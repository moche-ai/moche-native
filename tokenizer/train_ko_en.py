#!/usr/bin/env python3
"""G0-2: 한·영 64k 토크나이저 학습 + fertility 리포트.
- 데이터: KO 1B chars(fineweb2-edu-korean-score2) + EN 1B chars(ClimbMix) 교대 혼합, 문서당 10k cap
- 비교: ours-64k vs nanochat-32k(캘리브레이션용, EN 전용) vs GPT-2 vs GPT-4(cl100k)
- 산출: tokenizer/ko-en-64k/ + tokenizer/FERTILITY.md
실행: exp/nanochat 프로젝트 컨텍스트에서 (NANOCHAT_BASE_DIR 필요)
"""
import glob, os, sys, time

LAB = "."
sys.path.insert(0, os.path.join(LAB, "exp/nanochat"))
os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.join(LAB, "data/scratch/nanochat"))

import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

KO_FILES = sorted(glob.glob(os.path.join(LAB, "data/raw/ko/fineweb2-edu-korean-score2/**/*.parquet"), recursive=True))
DOC_CAP = 10_000
PER_LANG = 1_000_000_000  # 언어당 1B chars

def ko_docs(files):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=["text"]):
            for t in batch.column("text").to_pylist():
                if t:
                    yield t

def en_docs():
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            yield doc

def mixed_iter():
    ko, en = ko_docs(KO_FILES[:-2]), en_docs()   # 마지막 2개 파일은 fertility 평가용 held-out
    n_ko = n_en = 0
    ko_done = en_done = False
    while not (ko_done and en_done):
        if not ko_done:
            try:
                d = next(ko)[:DOC_CAP]; n_ko += len(d); yield d
                ko_done = n_ko >= PER_LANG
            except StopIteration:
                ko_done = True
        if not en_done:
            try:
                d = next(en)[:DOC_CAP]; n_en += len(d); yield d
                en_done = n_en >= PER_LANG
            except StopIteration:
                en_done = True
    print(f"train chars — ko {n_ko:,} · en {n_en:,}")

OUT = os.path.join(LAB, "tokenizer", "ko-en-64k")
if not os.path.exists(os.path.join(OUT, "tokenizer.pkl")):
    t0 = time.time()
    tok = RustBPETokenizer.train_from_iterator(mixed_iter(), 65536)
    print(f"학습 {time.time()-t0:.1f}s")
    tok.save(OUT)
else:
    tok = RustBPETokenizer.from_directory(OUT)
    print("기존 학습본 로드")

# ── fertility 평가 (held-out) ──────────────────────────────────────────────
def sample_chars(gen, n=3_000_000):
    buf = []
    total = 0
    for d in gen:
        buf.append(d[:DOC_CAP]); total += min(len(d), DOC_CAP)
        if total >= n:
            break
    return "\n".join(buf)

ko_eval = sample_chars(ko_docs(KO_FILES[-2:]))
en_eval = sample_chars(en_docs())  # ClimbMix 앞쪽이지만 32k와 동일 조건 비교라 무방

import tiktoken
comps = {
    "ours ko-en-64k": tok,
    "nanochat-32k(EN)": RustBPETokenizer.from_directory(os.path.join(os.environ["NANOCHAT_BASE_DIR"], "tokenizer")),
    "GPT-2": tiktoken.get_encoding("gpt2"),
    "GPT-4(cl100k)": tiktoken.get_encoding("cl100k_base"),
}
rows = []
for name, t in comps.items():
    r = {}
    for lang, text in (("KO", ko_eval), ("EN", en_eval)):
        ids = t.encode(text) if not hasattr(t, "encode_ordinary") else t.encode_ordinary(text)
        r[lang] = len(text) / len(ids)   # chars per token (높을수록 효율적)
    rows.append((name, r["KO"], r["EN"]))

lines = ["# 한·영 64k 토크나이저 fertility 리포트 (G0-2)", "",
         f"- 학습: KO {PER_LANG/1e9:.0f}B + EN {PER_LANG/1e9:.0f}B chars · vocab 65,536 · held-out 평가(각 ~3M chars)",
         "", "| 토크나이저 | KO (자/토큰↑) | EN (자/토큰↑) |", "|---|---|---|"]
for name, ko_r, en_r in rows:
    lines.append(f"| {name} | {ko_r:.2f} | {en_r:.2f} |")
base_ko = rows[1][1]
lines += ["", f"- 한국어 개선: nanochat-32k {base_ko:.2f} → ours {rows[0][1]:.2f} (**{rows[0][1]/base_ko:.1f}×**)",
          f"- GPT-4 대비 한국어: {rows[0][1]/rows[3][1]:.2f}×"]
report = "\n".join(lines)
open(os.path.join(LAB, "tokenizer", "FERTILITY.md"), "w").write(report + "\n")
print(report)
