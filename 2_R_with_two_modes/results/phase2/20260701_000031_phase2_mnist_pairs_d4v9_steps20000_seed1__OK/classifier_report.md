# Phase 2 — φ Classifier Evaluation (test set)

DMSR_φ(λ)와 FID-φ 계열 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로, φ 자체의 분류 성능을 먼저 점검한다. (생성 샘플은 unconditional이라 정답이 없어 이 지표들은 **테스트셋**에서만 정의된다.)

- digits: 4, 9
- test samples: 1991
- **accuracy: 99.50%**
- macro avg — precision 0.9950, recall 0.9950, F1 0.9950
- weighted avg — precision 0.9950, recall 0.9950, F1 0.9950

## Per-class metrics

| digit | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| 4 | 0.9939 | 0.9959 | 0.9949 | 982 |
| 9 | 0.9960 | 0.9941 | 0.9950 | 1009 |

## Confusion matrix (행=정답 true, 열=예측 pred)

| true \\ pred | 4 | 9 |
|---|---|---|
| **4** | 978 | 4 |
| **9** | 6 | 1003 |

> precision=특정 숫자로 예측한 것 중 실제로 그 숫자인 비율, recall=실제 그 숫자인 것 중 맞춘 비율, F1=둘의 조화평균(불균형에 강건).
