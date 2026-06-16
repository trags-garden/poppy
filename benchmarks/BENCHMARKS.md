# Engine benchmarks

Poppy ships three retrieval engines, `bloom` (default), `sprout`, and `seed`.
This page shows how they compare on retrieval quality, latency, and footprint, so
you can pick the right tradeoff.

These are **retrieval** numbers (did the engine surface the memory that holds the
answer), not end-to-end answer accuracy, and they cover Poppy's own engines only.
A fuller, fully reproducible benchmark, including a comparison against other
memory systems, is a separate planned effort.

## Results

Measured on two public long-term-memory datasets, the first 200 questions of
each, single run. `recall_any@k` is the share of questions where a gold memory
lands in the top `k`; `nDCG@10` rewards ranking it higher.

**LongMemEval-S** (session-level retrieval):

| Engine | recall@1 | recall@5 | recall@10 | nDCG@10 |
|---|---|---|---|---|
| `seed` | 0.0% | 0.0% | 0.0% | 0.000 |
| `sprout` | 91.0% | 97.5% | 99.0% | 0.947 |
| `bloom` | 91.0% | 97.5% | 99.0% | 0.947 |

**LoCoMo** (turn-level retrieval, the harder setting):

| Engine | recall@1 | recall@5 | recall@10 | nDCG@10 |
|---|---|---|---|---|
| `seed` | 0.0% | 0.0% | 0.0% | 0.000 |
| `sprout` | 50.5% | 68.0% | 75.5% | 0.595 |
| `bloom` | 50.5% | 69.5% | 77.0% | 0.606 |

## Footprint and latency

| Engine | Dependencies | Model download | Retrieve latency |
|---|---|---|---|
| `seed` | none (standard library) | 0 MB | under 1 ms |
| `sprout` | sentence-transformers + torch, ~600 MB | ~180 MB | ~0.2 to 0.6 s |
| `bloom` | sentence-transformers + torch, ~600 MB | ~225 MB | ~0.2 to 0.6 s |

Footprint sizes are MB = 10^6 bytes for a CPU install; a default CUDA torch on
Linux adds roughly 2 GB of NVIDIA libraries. Latency is wall-clock per query,
retrieval only, on a CPU laptop; treat it as order of magnitude. The cross-
encoder rerank is what separates `seed` (sub-millisecond) from the ML engines.

## What these numbers mean

- **`seed` scores near zero here, by design of the test.** Both datasets ask
  natural-language questions and score retrieval at the level of a single turn or
  session, so pure keyword (FTS5) matching rarely surfaces the exact gold passage
  from a paraphrased question. `seed` is the zero-dependency, offline floor and
  the keyword layer inside `sprout` and `bloom`, not a verdict on everyday
  keyword lookups over your own notes. The gap is the reason Poppy ships an ML
  engine as the default.
- **`sprout` and `bloom` tie on LongMemEval.** Each question there has a small
  candidate set, so both engines feed the same candidates to the same cross-
  encoder. The stronger `bloom` bi-encoder only pulls ahead when the candidate
  set is large, as on LoCoMo (77.0% vs 75.5% at recall@10).
- **Single run.** Each cell is one pass over 200 questions, so treat the values
  as point estimates rather than precise figures.

## Datasets

Public datasets, downloaded from their upstream sources and not redistributed
here.

- **LongMemEval-S (cleaned)**, MIT. Wu et al. (2024), *LongMemEval: Benchmarking
  Chat Assistants on Long-Term Interactive Memory*, arXiv:2410.10813 (ICLR 2025).
- **LoCoMo-10**, CC BY-NC 4.0 (non-commercial). Maharana et al. (2024),
  *Evaluating Very Long-Term Conversational Memory of LLM Agents*,
  arXiv:2402.17753. Reported here for non-commercial research comparison only.
