# Phase 3.2 — Mixture Noise-Schedule Experiment

Phase 3의 재설계 버전. **기존 `phase3/`는 그대로 두고** 여기서 새 설계를 돌린다.
결과는 `results/phase3_2/`에 저장된다(기존 `results/phase3/`와 분리).

## 왜 재설계했나

Phase 3에서 좁게 집중한 schedule(예: Laplace b=0.5)이 FID 400+로 붕괴하고 uniform이
이기던 원인은 **support mismatch**였다: 학습은 λ_R*(=3.37, low-noise) 근처에 몰리는데,
DDIM 생성은 항상 high-noise(λ≈−8)에서 시작 → 생성 시작점이 미학습이라 오차가 누적.

Phase 3.2는 이를 원천 차단한다:

1. **학습 λ 범위 = 생성 λ 범위 = [−8, 8]** 로 일치 (`lambda_min/max`, `ddim_lambda_min/max`).
2. **Mixture 분포**로 coverage와 emphasis를 분리:

   ```
   p_train(λ) = (1 − η)·Uniform[−8,8]  +  η·TruncatedLaplace[−8,8](center, b)
   ```

   - **Uniform floor** → 생성 trajectory 전체(λ=−8 시작점 포함)를 항상 학습 → 붕괴 차단.
   - **Laplace peak** → DMSR 가설 검증. `center=λ_R*`(우리) vs `center=0`(Hang) 를
     같은 η·b에서 비교해 **'중심 위치 효과'만** 분리한다.
   - TruncatedLaplace는 **clamp 아님** — rejection resampling으로 진짜 truncate
     (경계 스파이크 없음). floor가 커버를 책임지므로 `b`는 작게(sharp) 잡아야 emphasis가 보인다.

## 고정 항목 (통제 변수 — EDM design-space separation)

VP ε-prediction · plain ε-MSE(손실 가중치 없음) · DDIM 고정 grid[−8,8] · CFG=1.5 ·
optimizer/steps/eval 전부 동일. **변수는 오직 `p_train(λ)`.**

## 실행

```bash
# 파이프라인 점검
bash run_phase3_2.sh smoke airplane_vs_frog

# Stage A — 탐색(seed 1): (η,b) sweep으로 좋은 조합 찾기
#   schedules = cosine_vp + uniform + {dmsr,at0}×{η∈0.5,0.75}×{b∈0.5,1.0} = 10
bash run_phase3_2.sh stageA airplane_vs_frog

# Stage B — 확정(seed 3): 고른 (η,b)로 중심 0 vs λ_R* 정면 비교 + baseline
#   schedules = cosine_vp + uniform + mix_dmsr + mix_at0 = 4  (×3 seed = 12 run)
bash run_phase3_2.sh stageB airplane_vs_frog 0.5 1.0

# 끊긴 실행 이어서
bash run_phase3_2.sh resume <folder> [max_new]
```

모듈 직접 호출도 가능:
```bash
python3 -m phase3_2.run --preset full --class-pair airplane_vs_frog \
    --num-seeds 3 --mixture-etas 0.5 --mixture-bs 1.0
```

## 결과 구조

각 실행은 `results/phase3_2/` 아래에 독립 폴더로 저장된다. `run_phase3_2.sh`를 쓰면
폴더명에 `smoke`, `stageA`, `stageB_eta..._b...`, class pair, step 수, seed 수가 들어간다.

```text
results/phase3_2/
  YYYYMMDD_HHMMSS_phase3_2_stageA_airplane_vs_frog_steps100000_seed1__OK/
    run_meta.json              # 실행 서버, CUDA_VISIBLE_DEVICES, git commit, 상태, 소요 시간
    config.json                # 실행 설정
    schedules.json             # 실제 비교한 schedules와 lambda_R* 중심
    summary.md                 # 사람이 먼저 읽을 요약
    metrics_summary.csv        # schedule x seed raw metrics
    metrics_aggregated.csv     # seed 집계
    per_lambda_metrics.csv     # lambda별 denoising MSE
    stats.json                 # 유의성/집계/진단 상세
    classifier_report.*        # phi classifier 점검
    sched_records/             # schedule별 완료 record; resume 때 재사용
    model_*.pt                 # schedule별 학습 모델
    plots/                     # DMSR, density, samples, metric plots
```

실행 중이면 `__RUNNING`, 정상 완료면 `__OK`, `--max-new` 등으로 일부만 끝나면 `__PARTIAL`,
실패하면 `__FAILED_<stage>`로 끝난다. `__PARTIAL` 또는 실패 폴더는 `resume`으로 이어서 돌릴 수 있다.

## 핵심 비교

| schedule | 의미 |
|---|---|
| `cosine_vp` | 관행 baseline (유의성 검정 기준) |
| `uniform` | 무정보 baseline (mixture η=0 특수형) |
| `mix_dmsr_eta{η}_b{b}` | peak center = **λ_R*** ← 우리 가설 |
| `mix_at0_eta{η}_b{b}` | peak center = **0** ← Hang-style 대조 |

**연구 질문:** 고정된 sampler·objective 하에서, mixture peak을 λ=0이 아니라 λ_R*에
두면 생성 성능(FID)이 개선되는가? → `mix_dmsr` vs `mix_at0` vs `uniform` 비교로 답한다.

## 코드 변경점 (phase3 대비)

- `config.py`: `lambda_min/max` → ±8, `mixture_etas`/`mixture_bs` 추가, `ScheduleSpec.eta` 추가.
- `experiment.py`: `_sample_truncated_laplace`(rejection), `sample_schedule`에 `"mixture"` kind,
  `_mixture_pdf`(정규화된 truncated 밀도), `build_schedules` mixture로 재작성.
- `run.py`: 출력 경로 `results/phase3_2/`, CLI `--mixture-etas/--mixture-bs`.
