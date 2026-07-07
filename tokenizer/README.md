---
license: mit
language:
- ko
- en
library_name: tokenizers
tags:
- tokenizer
- bpe
- korean
- bilingual
- from-scratch
---

# moche-tokenizer-ko-en-64k

**한국어+영어 균형 64k BPE 토크나이저** — [moche-native](https://github.com/moche-ai/moche-native) 프롬스크래치 프로젝트(G0-2)의 자체 학습 토크나이저.

밑바닥부터 만드는 한·영 소형 모델을 위해, 영어 전용 토크나이저의 한국어 비효율을 해결하려고 직접 학습했다.

## Fertility (자/토큰, 높을수록 효율적)

held-out 평가 (각 언어 ~3M chars):

| 토크나이저 | 한국어 | 영어 |
|---|---|---|
| **moche ko-en-64k** | **2.47** | 4.58 |
| nanochat-32k (EN) | 0.49 | 4.66 |
| GPT-2 | 0.50 | 4.66 |
| GPT-4 (cl100k) | 1.02 | 4.82 |

- **한국어 fertility가 GPT-2 대비 5.0×, GPT-4 대비 2.4×.**
- 영어는 64k 어휘를 두 언어가 나눠 쓰는 트레이드오프로 소폭(-3%) 양보 — 한국어 5배 개선의 대가로 명백히 이득.

## 학습 상세

- 알고리즘: BPE (rustbpe, GPT-4 스타일 pre-tokenization)
- vocab: 65,536
- 데이터: 한국어 1B chars ([fineweb-2-edu-korean-score-2](https://huggingface.co/datasets/minpeter/fineweb-2-edu-korean-score-2)) + 영어 1B chars ([ClimbMix](https://huggingface.co/datasets/nvidia/ClimbMix)) 교대 혼합, 문서당 10k chars cap
- 학습 시간: ~3분 (CPU)
- 평가: held-out 문서 (학습에 미사용)

## 사용

```python
# nanochat RustBPETokenizer 포맷 (tokenizer.pkl)
# 학습·평가 스크립트: https://github.com/moche-ai/moche-native/blob/main/tokenizer/train_ko_en.py
```

## 라이선스

MIT. 학습 데이터는 전부 공개 코퍼스(ClimbMix, fineweb-2-edu-korean).

---

*moche-native — 첫 토큰부터 직접 만드는 한·영 옴니 모델. [프로젝트 저장소](https://github.com/moche-ai/moche-native)*
