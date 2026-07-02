# Phase 3 — φ Classifier Evaluation (test set)

DMSR_φ·FID-φ 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로 φ 자체의 분류 성능을 점검한다.
(생성 샘플은 unconditional이라 정답이 없어 이 지표들은 테스트셋에서만 정의된다.)

- classes: airplane, frog
- test samples: 2000
- **accuracy: 98.15%**
- macro avg — precision 0.9815, recall 0.9815, F1 0.9815

## Per-class metrics

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| airplane | 0.9801 | 0.9830 | 0.9815 | 1000 |
| frog | 0.9829 | 0.9800 | 0.9815 | 1000 |

## Confusion matrix (행=정답 true, 열=예측 pred)

| true \\ pred | airplane | frog |
|---|---|---|
| **airplane** | 983 | 17 |
| **frog** | 20 | 980 |
