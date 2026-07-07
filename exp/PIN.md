# exp/nanochat 핀 (재현성 스탬프)
- upstream: https://github.com/karpathy/nanochat
- commit: 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
- cloned: 2026-07-07
- 용도: G0 캘리브레이션(d20 재현→실측 tok/s·MFU) + 실험 트랙 골격

## 로컬 패치 (sm_120 = RTX PRO 6000 Blackwell 대응, 2026-07-07)
1. `nanochat/flash_attention.py` — `NANOCHAT_ATTN` 환경변수 오버라이드 추가.
   원인: kernels 허브의 FA3에 sm_120 바이너리가 없는데 has_kernel()이 오탐 → 런타임 "no kernel image" 크래시.
   해법: `NANOCHAT_ATTN=sdpa` 강제 폴백 (+ SDPA는 슬라이딩윈도 미지원 → `--window-pattern=L`).
2. `nanochat/common.py` — peak FLOPS 표에 RTX PRO 6000 (Max-Q=439e12, 표준=500e12) 추가 (MFU 계산용).
   ※ 실측 MFU 44.8%로 보아 SDPA(내장 flash 백엔드)가 sm_120에서 충분히 효율적.
3. `scripts/base_train.py` — wandb.init 프로젝트명을 `WANDB_PROJECT` env로 지정 가능하게 (기본값 업스트림 동일).
