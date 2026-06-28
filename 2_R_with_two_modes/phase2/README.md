# Phase 2: MNIST Pilot — DMSR Pipeline & 통계 틀 검증

이 단계의 목적은 VP diffusion 학습 코드, schedule sampler, empirical DMSR_φ 계산 파이프라인, DDIM 샘플러, FID 측정, **seed 간 통계 집계·유의성 검정**, visualization 전체가 올바르게 작동하는지 확인하는 것입니다. MNIST는 너무 쉬운 데이터셋이므로 schedule 간 FID 차이가 작게 나와도 실패가 아닙니다. **이 단계의 목표는 구현·통계 틀 검증**이며, 여기서 검증한 도구를 Phase 3(CIFAR 본 실험)에서 그대로 사용합니다.

## 코드 구성 (모듈별 역할)

| 파일 | 역할 |
|---|---|
| `config.py` | `ExperimentConfig`(모든 하이퍼파라미터), `ScheduleSpec`(p_train(λ) 명세) |
| `models.py` | 데이터 로딩, Mini U-Net(denoiser), Feature CNN φ, VP α/σ 유틸 |
| `experiment.py` | empirical DMSR_φ 계산, schedule 샘플러, 학습, DDIM, 평가 |
| `run.py` | 플로팅, 산출물 저장, 메인 실험 러너, CLI |
| `../stats_analysis.py` | **(공용)** seed 간 집계 + paired 유의성 검정 |
| `../gen_metrics.py` | **(공용)** FID·KID·Precision/Recall/Density/Coverage |
| `../gpu_perf.py` | **(공용)** GPU 가속(AMP·TF32·cudnn·torch.compile) 설정 |
| `../ddim_grid.py` | **(공용)** DDIM 샘플링 λ 격자(cosine) — 두 phase 동일 |

> `stats_analysis.py` 와 `gen_metrics.py` 는 `2_R_with_two_modes/` 최상위에 있는
> **공용 모듈**로, Phase 2와 Phase 3가 **동일한 코드**로 통계·평가를 수행하도록 한다.
> 덕분에 "Phase 2에서 검증한 도구를 Phase 3에 그대로 적용한다"는 논리가 코드 수준에서
> 성립한다.

## 평가 지표 (Phase 3와 동일한 정의)

생성 품질은 φ-feature 공간에서 다음을 모두 측정한다(`gen_metrics`). FID 하나로는
품질과 다양성이 섞이기 때문에 분리해서 본다.

| 지표 | 의미 | 방향 |
|---|---|---|
| FID (φ) | 두 분포의 평균·공분산 차이 | 낮을수록 좋음 |
| KID (φ) | 다항 커널 MMD². 표본 적을 때 FID보다 신뢰성 높음 | 낮을수록 좋음 |
| Precision (φ) | 생성 샘플이 진짜 manifold 안에 든 비율 (**품질**) | 높을수록 좋음 |
| Recall (φ) | 진짜 샘플이 생성 manifold 안에 든 비율 (**다양성**) | 높을수록 좋음 |
| Density / Coverage (φ) | Precision/Recall의 robust 개선판 | 높을수록 좋음 |

Phase 3는 위 φ 지표 일습에 더해 **InceptionV3 기반 FID**(CIFAR 표준)를 헤드라인으로 함께 보고한다.

## 실행 방법

### 패키지 설치 (최초 1회)

```bash
pip install torch torchvision scipy matplotlib numpy
```

### 전체 실험 (기본 설정: 20k steps, FID 5k, DDIM 50 steps)

```bash
python3 phase2/phase2_mnist_experiment.py
```

### GPU 서버 실행 (예: RTX 4090)

`--device auto`(기본)이면 CUDA를 자동 사용하고, 다음 GPU 가속이 **자동으로 켜집니다**:
혼합정밀(bf16 AMP), TF32, cudnn autotuner, `torch.compile`, 큰 생성 배치. 모든 schedule에
동일하게 적용되므로 비교 공정성은 유지됩니다.

```bash
# 기본(권장): bf16 AMP + compile 자동
python3 phase2/phase2_mnist_experiment.py --device cuda

# 정확 재현이 필요하면 정밀도 고정 / compile 끄기
python3 phase2/phase2_mnist_experiment.py --device cuda --amp fp32 --no-compile
```

- `--amp {auto,bf16,fp16,fp32}` : 기본 `auto`(CUDA면 bf16). `fp32`면 가속 끔.
- `--no-compile` : `torch.compile`이 환경과 안 맞을 때 비활성화(자동 폴백도 있음).
- `--gen-batch-size` : 생성 배치(기본 500). VRAM 여유가 크면 더 키워도 됨.

### 스모크 테스트 (빠른 동작 확인)

```bash
python3 phase2/phase2_mnist_experiment.py \
    --train-steps 500 --n-generate 50 --ddim-steps 5 \
    --dmsr-grid-size 12 --eval-grid-size 12 --clf-epochs 2
```

### 다른 digit 쌍으로 실행

```bash
python3 phase2/phase2_mnist_experiment.py --digits 3 8
```

### GPU 사용 (있는 경우)

```bash
python3 phase2/phase2_mnist_experiment.py --device cuda
```

## Phase 2 설계

### DMSR 지표의 활용

MNIST digit 0 vs 1 데이터로 간단한 CNN classifier를 학습하고 penultimate feature φ(x)를 추출합니다. VP λ grid에서 `x_λ = α_λ x₀ + σ_λ ε`를 생성해 empirical DMSR_φ(λ)를 계산하고 λ_R*를 추정합니다.

```
DMSR_φ(λ) = ‖μ_{φ,A}(λ) − μ_{φ,B}(λ)‖ / √((tr(Σ_{φ,A}(λ)) + tr(Σ_{φ,B}(λ))) / 2)
```

이 λ_R*는 Phase 3의 CIFAR 실험에 사용되지 않습니다. Phase 3에서는 CIFAR 데이터로 독립적으로 λ_R*를 재추정합니다.

### 고정 / 변경 사항

| 항목 | 설정 |
|---|---|
| 데이터 | MNIST digit 0 and 1 |
| 이미지 크기 | 1 × 28 × 28 |
| Forward process | VP: x_λ = α_λ x₀ + σ_λ ε |
| 모델 | Mini U-Net (base_ch=32, 2 resolution levels). Epsilon-prediction. |
| Training steps | 20k (스모크 테스트: 500) |
| Batch size | 128 |
| Optimizer | AdamW, lr=2e-4 |
| Loss | w(λ) = 1, uniform MSE |
| Sampling (고정) | DDIM 50 steps (VP cosine). 모든 모델에서 동일. |
| FID | φ-feature space로 계산 (5k samples) |

### 비교 Schedule (Phase 3와 동일하게 유지)

변경되는 것은 `p_train(λ)` 하나뿐이며, 나머지(loss weighting, sampler, 모델 구조, optimizer, steps)는 전부 고정합니다(EDM식 통제 설계).

- `cosine_vp`: VP cosine schedule이 유도하는 λ 분포 (관행 baseline)
- `hang_laplace_b0.5`: λ=0 중심 Laplace (Hang et al. baseline)
- `dmsr_normal_s1.5` / `dmsr_normal_s0.8` / `dmsr_normal_s0.3`: λ_R* 중심 Normal, 폭 s sweep
- `dmsr_laplace_b0.5`: λ_R* 중심 Laplace (분포 형태 변형 비교군)

`--s-values`, `--laplace-b`, `--baseline-schedule` 로 조정할 수 있습니다.

### 통계 분석 프레임워크 (유의성 검증의 틀)

Phase 3 본 실험에서 "schedule 간 차이가 통계적으로 유의한가?"를 검증하기 위한 틀을 Phase 2에서 미리 구축·검증합니다. 핵심 설계는 다음과 같습니다.

- **Paired 설계**: 동일한 `seed_idx` 안에서 모든 schedule이 같은 `run_seed`를 공유하므로, schedule 간 비교가 대응표본(paired)이 됩니다.
- **집계** (`aggregate_over_seeds`): schedule별로 seed에 걸쳐 mean ± std ± sem ± n을 계산 → `metrics_aggregated.csv`.
- **유의성 검정** (`significance_tests`): baseline 대비 seed별 차이 Δ에 paired t-test(seed≥2), Wilcoxon(seed≥5)을 적용 → `significance.md`. seed가 부족하면 안전하게 건너뜁니다.
- **per-λ 곡선 집계** (`aggregate_per_lambda`): denoising MSE 곡선을 λ마다 seed 평균/표준편차로 요약 → `stats.json`.

> seed 1회 실행에서는 분산을 추정할 수 없어 유의성 검정이 비활성화됩니다. 의미 있는 검정은 `--num-seeds 3` 이상을 권장합니다.

```bash
# 통계 틀까지 함께 검증하려면 (느림)
python3 phase2/phase2_mnist_experiment.py --num-seeds 3
```

### 기대하는 결과

- 파이프라인 전체가 오류 없이 실행된다.
- empirical DMSR_φ(λ)가 λ 감소에 따라 단조감소하는 곡선으로 나온다.
- λ_R*가 추정되고, 해당 λ에서 noisy image가 시각적으로 구분하기 어려운 수준임을 확인한다.

## 저장 구조

```
results/phase2/<timestamp>_phase2_mnist_d0v1/
  config.json
  schedules.json
  dmsr_info.json           # empirical DMSR_φ(λ) grid + λ_R*
  classifier_phi.pt        # 학습된 φ classifier weights
  train_history.json
  metrics_summary.csv          # per-(schedule, seed) raw 결과
  metrics_aggregated.csv       # seed 간 집계 (mean/std/sem/n)
  per_lambda_metrics.csv
  significance.md              # baseline 대비 paired 유의성 검정 결과
  stats.json                   # 집계 + 검정 + per-λ 곡선 집계 (기계 판독용)
  summary.md                   # per-run + 집계 + 유의성 + 해석 가이드
  plots/
    dmsr_profile.png                  # empirical DMSR_φ(λ) 곡선
    noisy_images_at_transition.png    # λ_R* 주변 noisy image 시각화
    schedule_densities.png            # p_train(λ) 분포 + T_R overlay
    per_lambda_mse.png                # schedule별 per-λ denoising MSE
    fid_summary.png                   # FID bar chart (per-run)
    fid_mean_std.png                  # FID mean ± std (seed 간 통계)
    coverage_vs_fid.png               # M coverage vs FID scatter
    samples_<schedule>.png            # schedule별 생성 이미지 grid
```

## 해석

Phase 2에서 보고 싶은 신호:
1. **empirical DMSR_φ(λ)**가 analytic Phase 1과 비슷한 형태로 나타나는지 (단조증가 + transition peak)
2. **λ_R*가 합리적인 범위**에 있는지 (MNIST 0 vs 1은 비교적 구분 쉬우므로 λ_R*가 낮은 편일 것)
3. **FID 차이가 작더라도** 파이프라인이 오류 없이 동작하면 성공

Phase 2 결과의 λ_R* 추정값은 Phase 3에서 s 후보 범위를 조정하는 데 참고됩니다.