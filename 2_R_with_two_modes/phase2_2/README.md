# Phase 2.2 — MNIST Pipeline Pilot

Phase 2.2는 기존 `phase2/`를 건드리지 않고 새 실험 설계에 맞춰 분리한 MNIST 파일럿이다.
결과는 `results/phase2_2/`에 저장된다.

## 목적

Phase 2.2는 schedule 간 생성 성능 우위를 주장하는 실험이 아니다. Phase 3.2 전에 아래
파이프라인이 한 번에 정상 작동하는지 확인한다.

- MNIST 0 vs 1 feature extractor phi 학습
- empirical DMSR_phi(lambda) 계산
- lambda_R* 추정
- Phase 3.2와 같은 mixture sampler 표집
- diffusion denoiser 학습
- 고정 [-8,8] DDIM grid 생성
- FID_phi, KID_phi, classifier confidence, balance, sample grid 저장

## Schedule Set

기본 schedule은 4개다.

| schedule | p_train(lambda) | 목적 |
|---|---|---|
| `uniform` | `U[-8,8]` | 전체 lambda 구간 coverage 확인 |
| `cosine_vp` | cosine VP induced density, clipped to `[-8,8]` | baseline 구현 확인 |
| `mix_at0_eta0.5_b2` | `(1-eta)U + eta TruncatedLaplace(center=0,beta)` | Hang-style center pilot |
| `mix_dmsr_eta0.5_b2` | `(1-eta)U + eta TruncatedLaplace(center=lambda_R*,beta)` | DMSR center pilot |

기본값은 `eta=0.5`, `beta=2.0`이며 sweep하지 않는다. Truncated Laplace는 clamp가 아니라
rejection sampling으로 진짜 `[-8,8]` truncation을 사용한다.

## 실행

```bash
# 빠른 파이프라인 점검
bash run_phase2_2.sh smoke

# 기본 pilot: MNIST 0 vs 1, 4 schedules, single seed
bash run_phase2_2.sh pilot

# paired aggregation 코드까지 점검하고 싶을 때
bash run_phase2_2.sh paired
```

직접 실행:

```bash
python3 phase2_2/phase2_mnist_experiment.py --device cuda --amp bf16
```

## 결과 구조

각 실행은 `results/phase2_2/` 아래에 독립 폴더로 저장된다.

```text
results/phase2_2/
  YYYYMMDD_HHMMSS_phase2_2_mnist_pilot_d0v1_steps20000_seed1__OK/
    run_meta.json              # 실행 서버, GPU, git commit, 상태, 소요 시간
    config.json                # 실행 설정
    schedules.json             # 실제 비교한 schedules
    summary.md                 # 사람이 먼저 읽을 요약
    metrics_summary.csv        # schedule x seed raw metrics
    metrics_aggregated.csv     # seed 집계
    per_lambda_metrics.csv     # lambda별 denoising MSE
    stats.json                 # 집계/진단 상세
    plots/                     # DMSR, schedule density, sample grid, metric plots
```

실행 중이면 폴더명이 `__RUNNING`, 정상 완료면 `__OK`, 실패하면 `__FAILED_<stage>`로 끝난다.

## Scope

Phase 2.2 결과는 Phase 3.2의 통계적 근거로 사용하지 않는다. 여기서는 모든 artifact가
정상 저장되고, loss가 발산하지 않으며, generated sample이 대략 MNIST digit처럼 나오는지만
확인한다.
