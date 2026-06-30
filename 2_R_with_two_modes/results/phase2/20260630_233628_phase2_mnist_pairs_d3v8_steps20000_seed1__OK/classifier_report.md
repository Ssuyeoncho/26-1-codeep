# Phase 2 — φ Classifier Evaluation (test set)

DMSR_φ(λ)와 FID-φ 계열 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로, φ 자체의 분류 성능을 먼저 점검한다. (생성 샘플은 unconditional이라 정답이 없어 이 지표들은 **테스트셋**에서만 정의된다.)

- digits: 3, 8
- test samples: 1984
- **accuracy: 99.85%**
- macro avg — precision 0.9985, recall 0.9985, F1 0.9985
- weighted avg — precision 0.9985, recall 0.9985, F1 0.9985

## Per-class metrics

| digit | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| 3 | 0.9990 | 0.9980 | 0.9985 | 1010 |
| 8 | 0.9979 | 0.9990 | 0.9985 | 974 |

## Confusion matrix (행=정답 true, 열=예측 pred)

| true \\ pred | 3 | 8 |
|---|---|---|
| **3** | 1008 | 2 |
| **8** | 1 | 973 |

> precision=특정 숫자로 예측한 것 중 실제로 그 숫자인 비율, recall=실제 그 숫자인 것 중 맞춘 비율, F1=둘의 조화평균(불균형에 강건).
