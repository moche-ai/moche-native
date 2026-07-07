# 한·영 64k 토크나이저 fertility 리포트 (G0-2)

- 학습: KO 1B + EN 1B chars · vocab 65,536 · held-out 평가(각 ~3M chars)

| 토크나이저 | KO (자/토큰↑) | EN (자/토큰↑) |
|---|---|---|
| ours ko-en-64k | 2.47 | 4.58 |
| nanochat-32k(EN) | 0.49 | 4.74 |
| GPT-2 | 0.50 | 4.66 |
| GPT-4(cl100k) | 1.02 | 4.82 |

- 한국어 개선: nanochat-32k 0.49 → ours 2.47 (**5.0×**)
- GPT-4 대비 한국어: 2.42×
