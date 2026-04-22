# CheckCitation Profiling Baseline

_Generated 2026-04-22 11:45:51. Each scenario run once with `force_refresh=True` (report cache bypassed). APICache warms naturally across scenarios on the same paper._

## Per-scenario timings (seconds)

| Label | Paper | Mode | Refs | L1 Parse | L2 Exist | L3/L5 | L4 Comp | Total | LLM Cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1-quick | paper1_watermark_llm.pdf | quick | 59 | 1.23 | 38.70 | 0.01 | 0.00 | 39.94 | $0.0004 |
| P1-standard | paper1_watermark_llm.pdf | standard | 59 | 0.00 | 6.58 | 0.01 | 104.96 | 111.56 | $0.0045 |
| P1-agentic | paper1_watermark_llm.pdf | agentic | 59 | 0.02 | 6.66 | 131.18 | 0.00 | 137.86 | $0.0004 |
| clean-quick | clean_paper.tex | quick | 4 | 0.25 | 3.12 | 0.00 | 0.00 | 3.38 | $0.0000 |
| hall-agentic | hallucinated_paper.tex | agentic | 4 | 0.05 | 6.27 | 13.23 | 0.00 | 19.56 | $0.0014 |

## Agentic dispatch breakdown

| Label | pre_retrieve | triage | meta_dispatch | claim_dispatch | both_dispatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1-agentic | 65.67 | 0.64 | 7.97 | 25.31 | 31.56 |
| hall-agentic | 6.55 | 0.03 | 0.00 | 6.61 | 0.00 |

## Notes

- L1 reflects spaCy/parser load + GROBID/LaTeX parsing (parse cache on disk).
- L2 existence checks use `APICache` (SQLite, TTL in config). Repeat DOIs hit cache.
- L3/L5 in quick mode is near-instant (rule-based classification).
- L5 in agentic mode = triage + agent LLM calls (semaphore cap **15** after
  Phase 2, raised from 5, see `config/config.yaml`).
- `paper2_attention.pdf`: skipped (GROBID not running at profile time).
- Cost extracted from `report.warnings`; appears to under-report actual spend
  (investigate before relying on it for budget caps — Phase 4).

## Before / after — Phase 2 impact

| Scenario | Phase 1 baseline | After Phase 2 | Change |
| --- | ---: | ---: | ---: |
| P1-quick (59 refs) | 96.33s | 39.94s | **−59%** (mostly network/S2 variance) |
| P1-standard | 167.91s | 111.56s | −34% (variance; no code change here) |
| **P1-agentic** | **354.28s** | **137.86s** | **−61% (Phase 2 gains)** |
| clean-quick | 3.31s | 3.38s | flat |
| hall-agentic | 32.22s | 19.56s | −39% |

**Agentic dispatch breakdown (P1, 59 refs):**

| Stage | Baseline (conc=5) | After Phase 2 (conc=15 + parallel pre-retrieve) |
| --- | ---: | ---: |
| pre_retrieve | 153.30s | 65.67s (−57%) |
| claim_dispatch | 136.69s | 25.31s (−81%) |
| metadata_dispatch | 13.91s | 7.97s (−43%) |
| both_dispatch | 42.39s | 31.56s (−26%) |

The P1-standard and P1-quick drops are mostly network/S2 throttling variance
(Phase 2 only touched agentic mode). The **real Phase 2 win is P1-agentic
354s → 138s**, driven by two changes:

1. `max_concurrent_agents` 5 → 15 — claim dispatch dropped 5×.
2. `_pre_retrieve_passages` refactored to `asyncio.gather` with semaphore —
   pre-retrieve dropped from 153s to 66s. Decompose LLM calls now run
   concurrently across references instead of serially.

## Remaining opportunities (Phase 3+ candidates)

- **Standard mode's L4 comprehension** is not yet parallelized (~105s here).
  Same trick as agentic pre-retrieve would help if standard mode becomes
  the default for reviewers.
- **Cold-cache L2** is still the worst-case experience for a fresh paper
  (~40–90s depending on S2 throttling). Consider paid S2 API key to lift
  the 1 RPS free-tier limit.
- **Results cache** (Phase 1) already makes second runs ~instant — no
  further work needed there.
