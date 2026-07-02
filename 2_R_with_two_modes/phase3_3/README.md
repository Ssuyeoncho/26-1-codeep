# Phase 3.3 — ImageNette-64 Mixture Schedule Experiment

Phase 3.3은 기존 `phase3_2/`를 건드리지 않고, ImageNet-derived natural image subset인
ImageNette를 64x64로 낮춰 같은 schedule-center 가설을 점검하는 확장 실험이다.
결과는 `results/phase3_3/`에 저장된다.

## 목적

이 실험은 full ImageNet 고품질 생성을 목표로 하지 않는다. 목표는 더 자연 이미지다운
two-class setting에서, 동일한 model/objective/sampler 아래 training noise distribution의
peak center를 어디에 두는지가 상대적으로 어떤 차이를 만드는지 보는 것이다.

핵심 질문:

```text
center = lambda_R* (DMSR transition) 가 center = 0 (Hang-style) 보다 나은가?
```

## 데이터

기본 데이터는 `imagenette2-160`이며, 로더는 아래 구조를 기대한다.

```text
data/imagenette2-160/
  train/<synset>/*.JPEG
  val/<synset>/*.JPEG
```

폴더가 없으면 코드가 fastai ImageNette tarball 다운로드를 시도한다. 다만 학교 서버에서
외부 다운로드가 느릴 수 있으므로, 실제 운용은 로컬에서 받은 `imagenette2-160.tgz` 또는
압축 해제 폴더를 `data/` 아래에 rsync로 올리는 것을 권장한다.

모든 이미지는 `Resize(64) + CenterCrop(64)` 후 `[-1,1]`로 정규화한다.

## 추천 class pair

| pair | 용도 |
|---|---|
| `gas_pump_vs_golf_ball` | 1순위. 구조/색감 차이가 커서 sanity check에 좋음 |
| `church_vs_garbage_truck` | 자연 이미지 구조가 더 풍부한 쉬운 pair |
| `english_springer_vs_garbage_truck` | 생물 vs 물체, 다만 dog variation이 있음 |
| `tench_vs_english_springer` | 어려운 생물 pair, 진단용 |
| `cassette_player_vs_chain_saw` | 복잡한 물체 pair, 진단용 |

## Schedule Set

Phase 3.2와 같은 mixture family를 사용한다.

```text
p_train(lambda) =
  (1 - eta) * Uniform[-8,8]
  + eta * TruncatedLaplace[-8,8](center, b)
```

기본 비교:

| schedule | 의미 |
|---|---|
| `cosine_vp` | 관행 baseline |
| `uniform` | full-range coverage baseline |
| `mix_at0_eta{eta}_b{b}` | center = 0, Hang-style 대조군 |
| `mix_dmsr_eta{eta}_b{b}` | center = lambda_R*, DMSR 중심 |

## 모델/학습

- image size: 64x64
- U-Net: CIFAR용 구조를 바탕으로 하되 64x64에서 attention이 16x16 쪽에 오도록 조정
- objective: plain epsilon-MSE
- sampler: DDIM 50 steps, lambda grid `[-8,8]`
- CFG: 1.5
- default batch size: 64
- default micro batch size: 16

## 실행

```bash
# 빠른 파이프라인 점검
bash run_phase3_3.sh smoke gas_pump_vs_golf_ball

# Stage A: eta/b sweep, single seed
bash run_phase3_3.sh stageA gas_pump_vs_golf_ball

# Stage B: 고른 eta/b로 seed 3개 확정 비교
bash run_phase3_3.sh stageB gas_pump_vs_golf_ball 0.5 1.0

# 끊긴 실행 이어서
bash run_phase3_3.sh resume <folder> [max_new]
```

직접 실행:

```bash
python3 -m phase3_3.run --preset fast --class-pair gas_pump_vs_golf_ball \
  --mixture-etas 0.5 0.75 --mixture-bs 0.5 1.0
```

## 결과 구조

```text
results/phase3_3/
  YYYYMMDD_HHMMSS_phase3_3_stageA_gas_pump_vs_golf_ball_steps50000_seed1__OK/
    run_meta.json
    config.json
    schedules.json
    summary.md
    metrics_summary.csv
    metrics_aggregated.csv
    per_lambda_metrics.csv
    stats.json
    classifier_report.*
    sched_records/
    model_*.pt
    plots/
```

`run_meta.json`에는 hostname, CUDA_VISIBLE_DEVICES, command, git branch/commit, 시작/종료
시각이 들어간다.

## 해석 범위

생성물이 ImageNet 논문 수준으로 선명할 것을 기대하면 안 된다. 이 실험에서 볼 것은:

- phi classifier가 ImageNette two-class를 충분히 구분하는지
- empirical DMSR_phi(lambda)가 안정적으로 계산되는지
- lambda_R*가 NaN 없이 추정되는지
- mixture sampler가 64x64 natural image setting에서도 학습을 망가뜨리지 않는지
- `mix_dmsr`와 `mix_at0`의 상대적 FID/phi-FID/precision/recall 차이가 일관적인지

즉 절대 생성 품질보다 **schedule center 위치 효과의 상대 비교**가 주 목적이다.
