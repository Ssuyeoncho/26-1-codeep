# results_from_server 안내 (무엇을, 어떤 의미로 봐야 하는가)

서버에서 돌린 실험 결과 모음입니다. 파일이 많아 헷갈릴 수 있어 정리했습니다.

## 1. 폴더 이름 읽는 법

각 실행은 `날짜_시각_이름` 형식의 폴더로 저장됩니다. 예:

```
20260628_183911_phase2_mnist_d0v1
└─날짜  └─시각 └이름     └두 클래스(digit 0 vs 1)
2026-06-28  18:39:11   phase2 MNIST
```

- `phase1_...` = 1D toy 실험 (2026-05-29에 돌린 옛 결과들)
- `phase2_...` = MNIST 실험 (2026-06-28에 돌린 것)

## 2. 지금 "봐야 할" 결과는 딱 하나

정리하면서 상태별로 나눠 두었습니다:

| 위치 | 내용 | 봐야 하나? |
|---|---|---|
| `phase2/20260628_183911_phase2_mnist_d0v1/` | **본 실행 (full)** — 20k steps, FID 5000장 | ✅ **이걸 보세요** |
| `phase2/_smoke_tests/` | 스모크 테스트 (500 steps, 동작 확인용) | 참고만 |
| `phase2/_aborted_runs/` | 중간에 멈춘 실행 (학습 결과 없음) | 무시 |
| `phase1/...` | 1D toy 옛 결과 7개 | 별개 단계 |

> `_aborted_runs`는 학습 전에 멈춰서 `summary.md`가 없습니다(아마 torch.compile 오류로 중단). 버리지 않고 모아만 뒀습니다.

## 3. full 폴더 안에서 — 어떤 파일을 어떤 의미로 보나

`phase2/20260628_183911_phase2_mnist_d0v1/` 기준입니다.

### 먼저 볼 것 (사람이 읽는 요약)
| 파일 | 의미 |
|---|---|
| **`summary.md`** | **제일 먼저.** 설정 + schedule별 결과표 + 해석 가이드 전부. |
| `significance.md` | schedule 간 차이의 통계적 유의성. *지금은 seed 1개라 비어 있음* (검정하려면 seed≥3). |

### 숫자 데이터 (엑셀로 열기)
| 파일 | 의미 |
|---|---|
| `metrics_summary.csv` | schedule별 핵심 지표(FID/KID/Precision/Recall/Density/Coverage/MSE…) |
| `metrics_aggregated.csv` | 위를 seed 간 평균±표준편차로 집계 (지금은 seed 1개) |
| `per_lambda_metrics.csv` | λ(노이즈 레벨)별 denoising MSE 곡선 raw 값 |
| `stats.json` | 위 통계 전체를 기계가 읽기 좋은 형태로 |
| `dmsr_info.json` | DMSR(λ) 곡선, λ_R*, transition 구간 값 |
| `config.json` | 이 실행에 쓴 모든 설정 |
| `schedules.json` | 비교한 6개 schedule 정의 |

### 그림 (`plots/` 폴더) — 보통 이걸 보면 직관적
| 파일 | 무엇을 보여주나 |
|---|---|
| `dmsr_profile.png` | **DMSR(λ) 곡선** + transition 구간 + λ_R*. 지표가 의도대로 작동하는지. |
| `schedule_densities.png` | 각 schedule이 λ를 **어디서 얼마나** 뽑는지(학습 분포). |
| `noisy_images_at_transition.png` | λ_R* 부근에서 0/1이 **눈으로 구분되는지** noisy 이미지. |
| `samples_<schedule>.png` | 각 schedule로 **생성한 숫자 이미지**(육안 품질). |
| `fid_summary.png` | schedule별 FID 막대 (낮을수록 좋음). |
| `fid_mean_std.png` | FID 평균±표준편차 (seed 여러 개일 때 의미; 지금은 막대만). |
| `per_lambda_mse.png` | schedule별 λ별 denoising MSE 곡선. |
| `coverage_vs_fid.png` | transition 덮은 정도(M) vs FID 산점도. |
| `classifier_phi.pt` | 학습된 φ 분류기 가중치(지표 계산용). 그림 아님. |

## 4. 이번 full 결과 한눈 요약 (digit 0 vs 1, seed 1)

- DMSR: λ_R* = **-0.26**, transition 구간 = **[-2.31, 1.28]** → 지표는 정상 작동.
- FID(낮을수록 좋음): `cosine_vp` **36.8** (1등) ≪ DMSR/Hang 계열(101~478).
- `dmsr_normal_s0.3`(가장 좁은 분포)은 FID 478 + **Precision 0 / Coverage 0** → **mode collapse**(생성 다양성 붕괴)를 지표가 정확히 잡아냄.

### 이 결과를 어떻게 받아들여야 하나 (중요)
- **이 단계(Phase 2)의 목적은 "성능 주장"이 아니라 "파이프라인·지표 검증"입니다.** 전 과정이 오류 없이 돌고, 지표가 합리적으로 반응(특히 좁은 분포의 붕괴를 Precision/Coverage가 0으로 잡음)하면 **성공**입니다. → 검증 성공.
- cosine_vp가 이긴 것에 큰 의미를 두지 마세요: **MNIST는 너무 쉽고, seed가 1개**라 통계적 결론을 낼 수 없습니다.
- 진짜 비교(유의성 포함)는 **Phase 3(CIFAR) + seed 3개 이상**에서 합니다. 그때 `significance.md`가 채워집니다.

## 5. 추천 보는 순서
1. `phase2/20260628_183911_.../plots/dmsr_profile.png` (지표 작동 확인)
2. 같은 폴더 `summary.md` (전체 표·해석)
3. `plots/samples_*.png` (생성 이미지 육안 비교 — 특히 s0.3의 붕괴)
4. `plots/schedule_densities.png` (각 schedule이 어디를 학습했나)
