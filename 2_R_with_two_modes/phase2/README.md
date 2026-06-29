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

### φ 분류기 평가 지표 (MNIST 분류 성능 — 정답 레이블 기반)

위 생성 품질 지표와 별개로, feature extractor φ(= MNIST 두 숫자 분류기) **자체의 분류
성능**을 테스트셋에서 보고한다(`../classification_metrics.py`, Phase 3와 공용). DMSR_φ(λ)와
모든 FID-φ 지표가 이 φ의 feature 위에서 계산되므로, φ가 충분히 잘 분류하는지를 먼저
점검해야 하위 지표들을 신뢰할 수 있다. (생성 샘플은 unconditional이라 정답이 없어, 이
지표들은 **테스트셋에서만** 정의된다.)

| 지표 | 의미 | 방향 |
|---|---|---|
| Accuracy | 전체 중 맞춘 비율 = (TP+TN)/(TP+TN+FP+FN) | 높을수록 좋음 |
| Confusion Matrix | 어떤 숫자를 어떤 숫자로 헷갈렸는지 (행=정답, 열=예측) | 대각선 집중이 좋음 |
| Precision | 그 숫자로 예측한 것 중 실제로 그 숫자인 비율 | 높을수록 좋음 |
| Recall | 실제 그 숫자인 것 중 맞춘 비율 | 높을수록 좋음 |
| F1 | Precision·Recall의 조화평균 (불균형에 강건) | 높을수록 좋음 |

→ `classifier_report.json` / `classifier_report.md` 로 저장되고 summary.md 에 요약된다.

### per-λ denoising MSE 분해 (noise level별 진단 — Phase 3와 공용)

per-λ ε-prediction MSE는 양 phase 모두 이미 측정한다(`per_lambda_mse`). 여기에 **모델
재실행 없이** 후처리만으로 두 가지 표준 view를 더해 p_train 효과를 noise level별로 분해한다
(`../stats_analysis.py: per_lambda_excess_and_skill`).

| 지표 | 정의 | 의미 |
|---|---|---|
| excess (vs baseline) | `MSE_s(λ) − MSE_cosine(λ)` | 같은 λ의 공통 Bayes floor가 차분에서 소거 → **순수 p_train 효과**. >0이면 그 noise 구간을 baseline보다 못 배움 |
| skill / R² | `1 − MSE/MSE_trivial` (ε-pred는 trivial=1) | U-Net이 각 구간에서 실제로 학습됐는지. 1=완벽, 0=trivial 예측과 동급, <0=더 나쁨 |

> **주의(honest scope):** Phase 1 toy처럼 analytic Bayes denoiser가 있는 경우의 *절대*
> excess MSE(model − Bayes)는 MNIST/CIFAR엔 최적 denoiser가 없어 계산 불가다. 그래서
> 여기서는 가짜 Bayes 기준을 만들지 않고, **baseline 차분(공통 floor 소거)** 으로만 excess를
> 정의한다. skill의 trivial 기준(=1)도 ε-prediction의 best constant predictor(0)에서 나온
> 사실 그대로다. 두 지표는 `metrics_summary.csv`(skill 스칼라)·`stats.json`
> (`per_lambda_diagnostics`)·`per_lambda_decomposition.png`·summary.md 에 기록된다.

## 실행 방법

### 패키지 설치 (최초 1회)

```bash
pip install torch torchvision scipy matplotlib numpy
```

### 전체 실험 (기본 설정: 20k steps, FID 5k, DDIM 50 steps)

```bash
python3 phase2/phase2_mnist_experiment.py
```

### GPU 서버 실행 (현재 서버 = RTX 4090 / Ada Lovelace, 24GB, CUDA 12.2)

가장 간단한 방법은 헬퍼 스크립트입니다(4090 효율 인자가 들어 있음):

```bash
bash run_phase2.sh smoke        # 빠른 동작 확인 (폴더명에 *_smoke 표시)
bash run_phase2.sh full         # 전체 (단일 seed, 기본 0 vs 1)
bash run_phase2.sh full 3 8     # 전체, 클래스 쌍 지정 (digit 3 vs 8)
bash run_phase2.sh final        # 최종 (seed 3개 — 통계·유의성)
bash run_phase2.sh pairs        # ★ 큐레이션된 여러 쌍을 한 번에 (무인 배치)
```

**4090(Ada Lovelace):** bf16/TF32를 하드웨어로 지원하므로 스크립트는 `--amp bf16`(fp16+scaler
보다 안정적)을 씁니다. VRAM 24GB라 MNIST Mini U-Net은 여유가 큽니다 — 학습 batch_size(=128)는
공정 비교 위해 유지(micro-batch=128, 분할 없음)하고 처리량용 생성 batch만 키웁니다
(`--gen-batch-size 1000`). 결과는 동일하고 속도만 오릅니다. OOM(거의 없음) 시 낮추세요.

#### 여러 클래스 쌍 한 번에 (무인 배치)

데이터(클래스 쌍)마다 DMSR 중심 λ_R*와 결과가 어떻게 달라지는지 보려면, 비슷/다름 난이도를
섞은 큐레이션 쌍을 한 번에 돌립니다. **한 쌍이 실패해도 멈추지 않고 다음 쌍으로 계속 진행**하며,
각 쌍은 자체 결과 폴더(`..._d3v8_..._OK` 등)에 저장되어 부분 완료도 보존됩니다.

```bash
bash run_phase2.sh pairs
#   쌍: 0v1, 1v8 (확연히 다름) / 3v8, 4v9 (비슷)
```

> Phase 2·3를 한 번에 다 돌리려면 최상위 `bash run_all_pairs.sh` (자기 전 1줄 실행). 로그아웃해도
> 계속 돌리려면 `nohup bash run_all_pairs.sh > run_all_pairs.log 2>&1 &`.

```bash
# 직접 실행 (스크립트 없이)
python3 phase2/phase2_mnist_experiment.py --device cuda --amp bf16 \
    --micro-batch-size 128 --gen-batch-size 1000 --digits 3 8
```

- `--amp {auto,bf16,fp16,fp32}` : 기본 `auto`(4090이면 bf16, 미지원 GPU면 fp16). `fp32`면 가속 끔.
- `--micro-batch-size` : GPU에 한 번에 올릴 학습 batch(전체 batch는 유지, 등가).
- `--gen-batch-size` : 생성 배치. VRAM 여유가 크면 더 키워도 됨.

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

**이름 규칙 — `<중심>_<분포형태>_<폭>`:**

**핵심 설계 — {Normal, Laplace} × {중심 0, 중심 λ_R\*} × {폭 sweep} 매칭 factorial:**
같은 모양·같은 폭에서 **중심만** 바꿔 나란히 둬서 "중심 위치(선행연구 0 vs 우리 λ_R\*) 효과"를 통제비교한다.

| 이름 | 중심 | 분포 | 폭 | 설명 |
|---|---|---|---|---|
| `cosine_vp` | λ≈0 | VP cosine-β (≈sech) | — | 관행 baseline (유의성 기준) |
| `linear_vp` | — | VP linear-β(DDPM) | — | 관행 baseline (옵션) |
| `uniform` | — | λ 범위 균일 | — | 무정보 baseline (옵션) |
| `dmsr_normal_s{w}` | **λ_R\*** | Normal | s=w | 우리: λ_R\* 중심 Normal |
| `at0_normal_s{w}` | **0** | Normal | s=w | 대조: 중심 0 Normal (같은 폭) |
| `dmsr_laplace_b{w}` | **λ_R\*** | Laplace | b=w | 우리: λ_R\* 중심 Laplace |
| `at0_laplace_b{w}` | **0** | Laplace | b=w | 대조: 중심 0 Laplace = **Hang et al. baseline** |
| `dmsr_studentt_s{w}` | λ_R\* | Student-t (ν=df) | scale=w | 꼬리가 끝까지 안 죽음(좁아도 전 구간 커버 + 집중) |
| ~~`dmsr_cosmix_w{w}`~~ | — | (구) cosine 혼합 | — | 기본 OFF (`--include-cosmix`). Student-t로 대체 |

**비교 축(읽는 법):** `dmsr_laplace_b1.5` ↔ `at0_laplace_b1.5` 처럼 **이름의 폭·모양이 같고 앞부분(dmsr/at0)만 다른 쌍**을 비교하면 "중심을 λ_R\*에 두는 것이 0보다 나은가?"가 바로 보인다. 폭은 `--width-values`로(기본 0.5/1.5/4.0, 모임↔퍼짐 확연히 차이), 중심 0 대조군 제외는 `--no-center0`.

> **cosmix → Student-t 교체(2026-06-30):** cosine을 외부에서 빌려 섞는 게 부자연스러워, λ_R\* 중심 단일 분포면서 꼬리가 다항식으로 천천히 죽는 **Student-t**로 바꿨다. Normal(꼬리 exp(−x²))은 좁히면 clean끝을 못 뽑아 붕괴하지만, Student-t(ν=3 기본)는 같은 좁은 scale에서도 |λ|>3 영역에 ~19% mass를 남겨(Normal은 ~0.3%) **집중과 전 구간 커버를 한 분포로** 달성한다. ν로 꼬리 두께 조절(1=Cauchy, ∞=Normal). cosine 의존 없음.

**왜 폭 기호가 b와 s로 다른가:** 둘 다 "퍼짐 정도"지만 **분포가 다릅니다**. Normal의 자연 모수는 표준편차 **s**(σ), Laplace의 자연 모수는 scale **b**(표준편차는 b√2). 같은 글자로 묶으면 같은 값이 같은 퍼짐을 뜻한다고 오해하므로 분포에 맞춰 다르게 씁니다.

**normal vs laplace 차이:** Normal은 꼬리가 가벼운(exp(−x²)) 종형, Laplace는 중심이 더 **뾰족**하고 꼬리가 더 **두꺼운**(exp(−|x|)) 형태입니다. 같은 중심에서 "분포 모양"이 결과에 주는 영향을 보려는 비교군입니다.

**cosine vs linear vs uniform:** 모두 λ_R* 정보를 **안 쓰는** 데이터 무관 기준입니다. cosine/linear는 각각 cosine-β/linear-β diffusion schedule이 유도하는 표준 분포이고, uniform은 전 구간 균일입니다. DMSR 기반 schedule이 이 관행 기준들을 이기는지 보는 비교축입니다.

> **왜 넓은 s·혼합·linear·uniform을 추가했나:** Phase 2 full에서 narrow-s(0.3·0.8)가 clean끝을 학습하지 못해 FID가 붕괴하고 cosine이 압승했다. 더 완만한(분산 큰) 분포로 "붕괴→회복"을 보려고 s를 6.0까지 넓히고, full-range support를 유지하는 cosine 혼합(cosmix)과, 관행 baseline 비교축(linear·uniform)을 추가했다. 변경되는 것은 여전히 `p_train(λ)` 하나뿐이다.

조정 인자: `--s-values`, `--laplace-b`, `--mix-weights`, `--mix-scale`, `--no-linear`, `--no-uniform`, `--baseline-schedule`.

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

폴더명에 **실행 시각 + 모드(steps·seed 수, smoke 여부) + 종료 상태**가 들어가 한눈에
구분됩니다. 실행 중에는 `__RUNNING`, 정상 종료 시 `__OK`, 중간에 실패하면 멈춘 단계를
붙여 `__FAILED_<stage>` 로 폴더가 자동 rename 됩니다. `run_meta.json`에는 시작/종료
시각·소요 시간·단계·에러 메시지가 기록됩니다(grep 으로 성공/실패 분류 가능).

```
results/phase2/<YYYYMMDD_HHMMSS>_<run_name>_d0v1_steps20000_seed1__OK/
  run_meta.json            # 시작/종료 시각·소요(초)·단계·상태·에러 (실행 추적)
  config.json
  schedules.json
  dmsr_info.json           # empirical DMSR_φ(λ) grid + λ_R*
  classifier_phi.pt        # 학습된 φ classifier weights
  classifier_report.json   # φ 분류기 테스트셋 평가 (accuracy/P/R/F1/confusion, 기계 판독용)
  classifier_report.md     # 위를 사람이 읽기 좋게 표로 정리
  train_history.json
  metrics_summary.csv          # per-(schedule, seed) raw 결과
  metrics_aggregated.csv       # seed 간 집계 (mean/std/sem/n)
  per_lambda_metrics.csv
  significance.md              # baseline 대비 paired 유의성 검정 결과
  stats.json                   # 집계 + 검정 + per-λ 곡선 집계 (기계 판독용)
  summary.md                   # per-run + 집계 + 유의성 + 해석 가이드
  plots/
    dmsr_profile.png                  # empirical DMSR_φ(λ) 곡선
    classifier_confusion_matrix.png   # φ 분류기 confusion matrix 히트맵 (행 정규화)
    classifier_pr_f1.png              # φ 분류기 클래스별 Precision/Recall/F1 막대
    noisy_images_at_transition.png    # λ_R* 주변 noisy image 시각화
    schedule_densities.png            # p_train(λ) 분포 + T_R overlay (경계 clamp 스파이크는 y축 자름)
    per_lambda_mse.png                # schedule별 per-λ denoising MSE
    per_lambda_decomposition.png      # per-λ MSE 분해: (위)baseline 대비 excess, (아래)skill=1−MSE
    lambda_learnability.png           # 의사결정용: λ별 학습난이도 + headroom(어디 더 뽑으면 좋을지)
    training_curves.png               # schedule별 학습 loss 곡선 (log-y, 수렴/발산 진단)
    coverage_vs_fid.png               # M coverage vs FID scatter
    precision_recall.png              # Precision(품질) vs Recall(다양성), mode-collapse 진단
    metric_overview.png               # 지표 종합 '표'(행=schedule, 열=지표, 지표별 best 수치 bold)
    samples_<schedule>.png            # schedule별 생성 이미지 grid
```

> **plot 점검(2026-06-29 갱신):**
> - **폰트 깨짐 방지:** 모든 plot 텍스트는 ASCII + 그리스문자(λ,φ,ρ)만 쓴다. matplotlib
>   기본 폰트(DejaVu Sans)에 한글 글리프가 없어 □로 깨지고 서버(Linux)에도 한글 폰트가
>   없을 수 있어, 그림 안 글자는 영어로 통일했다(요약 .md/터미널 출력은 한글 유지).
> - **schedule_densities는 이제 해석적(설계된) 밀도를 매끄러운 곡선으로** 그린다. 예전엔
>   80k 샘플 히스토그램이라 (1)계단형이고 (2)표본잡음 때문에 대칭 분포도 비대칭처럼 보였다.
>   닫힌 형태 밀도 p_train(λ)를 직접 평가하므로 매끄럽고 정확히 대칭이다.
> - **FID 전용 그림 2개(fid_summary, fid_mean_std)는 metric_overview에 흡수돼 삭제**했다.
> - **`lambda_learnability.png` 추가:** λ별 best-achievable MSE와 baseline의 차이(headroom)를
>   음영으로 보여 "어느 noise level을 더 뽑으면 개선 여지가 있는지" 의사결정을 돕는다.

## 클래스 쌍 바꿔 실행 (데이터별 λ_R* 비교)

데이터마다 DMSR 중심 λ_R*가 다르다는 가정을 다양한 two-class에서 보려면, 실행 시 클래스를
지정한다. (DMSR·λ_R*·schedule 중심이 모두 그 클래스 쌍 기준으로 자동 재계산된다.)

```bash
python3 phase2/phase2_mnist_experiment.py --digits 3 8        # 직접
bash run_phase2.sh full 3 8                                   # 헬퍼 스크립트 (digitA digitB)
```

## 터미널 요약

실행이 끝나면 핵심 결과가 콘솔에 한 화면으로 요약된다: λ_R*, φ 분류기 정확도, FID(φ) 순위
(best→worst, 각 schedule의 mean skill 포함), best vs baseline 비교, 결과 폴더 경로. 자세한
내용은 `summary.md`.

## 해석

Phase 2에서 보고 싶은 신호:
1. **empirical DMSR_φ(λ)**가 analytic Phase 1과 비슷한 형태로 나타나는지 (단조증가 + transition peak)
2. **λ_R*가 합리적인 범위**에 있는지 (MNIST 0 vs 1은 비교적 구분 쉬우므로 λ_R*가 낮은 편일 것)
3. **FID 차이가 작더라도** 파이프라인이 오류 없이 동작하면 성공

Phase 2 결과의 λ_R* 추정값은 Phase 3에서 s 후보 범위를 조정하는 데 참고됩니다.