# Phase 2 — φ Classifier Evaluation (test set)

DMSR_φ(λ)와 FID-φ 계열 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로, φ 자체의 분류 성능을 먼저 점검한다. (생성 샘플은 unconditional이라 정답이 없어 이 지표들은 **테스트셋**에서만 정의된다.)

- digits: 1, 8
- test samples: 2109
- **accuracy: 99.91%**
- macro avg — precision 0.9990, recall 0.9991, F1 0.9990
- weighted avg — precision 0.9991, recall 0.9991, F1 0.9991

## Per-class metrics

| digit | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| 1 | 1.0000 | 0.9982 | 0.9991 | 1135 |
| 8 | 0.9980 | 1.0000 | 0.9990 | 974 |

## Confusion matrix (행=정답 true, 열=예측 pred)

| true \\ pred | 1 | 8 |
|---|---|---|
| **1** | 1133 | 2 |
| **8** | 0 | 974 |

> precision=특정 숫자로 예측한 것 중 실제로 그 숫자인 비율, recall=실제 그 숫자인 것 중 맞춘 비율, F1=둘의 조화평균(불균형에 강건).
