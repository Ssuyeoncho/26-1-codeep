# Phase 3 — φ Classifier Evaluation (test set)

DMSR_φ·FID-φ 지표가 모두 이 분류기 φ의 feature 위에서 계산되므로 φ 자체의 분류 성능을 점검한다.
(생성 샘플은 unconditional이라 정답이 없어 이 지표들은 테스트셋에서만 정의된다.)

- classes: cat, dog
- test samples: 2000
- **accuracy: 83.35%**
- macro avg — precision 0.8336, recall 0.8335, F1 0.8335

## Per-class metrics

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| cat | 0.8400 | 0.8240 | 0.8319 | 1000 |
| dog | 0.8273 | 0.8430 | 0.8351 | 1000 |

## Confusion matrix (행=정답 true, 열=예측 pred)

| true \\ pred | cat | dog |
|---|---|---|
| **cat** | 824 | 176 |
| **dog** | 157 | 843 |
