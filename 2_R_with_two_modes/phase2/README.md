# Phase 2: MNIST Pilot — DMSR Pipeline Validation

이 단계의 목적은 VP diffusion 학습 코드, schedule sampler, empirical DMSR_φ 계산 파이프라인, DDIM 샘플러, FID 측정, visualization 전체가 올바르게 작동하는지 확인하는 것입니다. MNIST는 너무 쉬운 데이터셋이므로 schedule 간 FID 차이가 작게 나와도 실패가 아닙니다. **이 단계의 목표는 구현 검증**입니다.

## 실행 방법

### 패키지 설치 (최초 1회)

```bash
pip install torch torchvision scipy matplotlib numpy
```

### 전체 실험 (기본 설정: 20k steps, FID 5k, DDIM 50 steps)

```bash
python3 phase2/phase2_mnist_experiment.py
```

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

### 비교 Schedule

- `cosine_vp`: VP cosine schedule이 유도하는 λ 분포
- `hang_laplace_b0.5`: Hang-style Laplace centered at λ=0 (b=0.5)
- `dmsr_normal_wide_s1.5`: DMSR-centered Normal, s=1.5
- `dmsr_normal_mid_s0.8`: DMSR-centered Normal, s=0.8
- `dmsr_normal_narrow_s0.3`: DMSR-centered Normal, s=0.3

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
  metrics_summary.csv
  per_lambda_metrics.csv
  summary.md
  plots/
    dmsr_profile.png                  # empirical DMSR_φ(λ) 곡선
    noisy_images_at_transition.png    # λ_R* 주변 noisy image 시각화
    schedule_densities.png            # p_train(λ) 분포 + T_R overlay
    per_lambda_mse.png                # schedule별 per-λ denoising MSE
    fid_summary.png                   # FID bar chart
    coverage_vs_fid.png               # M coverage vs FID scatter
    samples_<schedule>.png            # schedule별 생성 이미지 grid
```

## 해석

Phase 2에서 보고 싶은 신호:
1. **empirical DMSR_φ(λ)**가 analytic Phase 1과 비슷한 형태로 나타나는지 (단조증가 + transition peak)
2. **λ_R*가 합리적인 범위**에 있는지 (MNIST 0 vs 1은 비교적 구분 쉬우므로 λ_R*가 낮은 편일 것)
3. **FID 차이가 작더라도** 파이프라인이 오류 없이 동작하면 성공

Phase 2 결과의 λ_R* 추정값은 Phase 3에서 s 후보 범위를 조정하는 데 참고됩니다.