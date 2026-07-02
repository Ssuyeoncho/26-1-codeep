# Phase 3 — φ Classifier Evaluation (test set)

DMSR_φ·FID-φ 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로 φ 자체의 분류 성능을 점검한다.
(생성 샘플은 unconditional이라 정답이 없어 이 지표들은 테스트셋에서만 정의된다.)

- classes: cat, dog
- test samples: 2000
- **accuracy: 84.00%**
- macro avg — precision 0.8402, recall 0.8400, F1 0.8400

## Per-class metrics

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| cat | 0.8476 | 0.8290 | 0.8382 | 1000 |
| dog | 0.8327 | 0.8510 | 0.8417 | 1000 |

## Confusion matrix (행=정답 true, 열=예측 pred)

| true \\ pred | cat | dog |
|---|---|---|
| **cat** | 829 | 171 |
| **dog** | 149 | 851 |
