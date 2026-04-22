# CheckCitation Profiling Baseline

_Generated 2026-04-22 08:16:42. Each scenario run once with `force_refresh=True` (report cache bypassed). APICache warms naturally across scenarios on the same paper._

## Per-scenario timings (seconds)

| Label | Paper | Mode | Refs | L1 Parse | L2 Exist | L3/L5 | L4 Comp | Total | LLM Cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1-quick | paper1_watermark_llm.pdf | quick | 59 | 4.07 | 92.23 | 0.02 | 0.00 | 96.33 | $0.0004 |
| P1-standard | paper1_watermark_llm.pdf | standard | 59 | 0.01 | 4.99 | 0.01 | 162.90 | 167.91 | $0.0045 |
| P1-agentic | paper1_watermark_llm.pdf | agentic | 59 | 0.01 | 6.01 | 348.27 | 0.00 | 354.28 | $0.0004 |
| clean-quick | clean_paper.tex | quick | 4 | 0.19 | 3.12 | 0.00 | 0.00 | 3.31 | $0.0000 |
| hall-agentic | hallucinated_paper.tex | agentic | 4 | 0.09 | 4.41 | 27.71 | 0.00 | 32.22 | $0.0014 |

## Agentic dispatch breakdown

| Label | pre_retrieve | triage | meta_dispatch | claim_dispatch | both_dispatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1-agentic | 153.30 | 1.86 | 13.91 | 136.69 | 42.39 |
| hall-agentic | 15.43 | 0.07 | 0.00 | 12.09 | 0.00 |

## Notes

- L1 reflects spaCy/parser load + GROBID/LaTeX parsing (parse cache on disk).
- L2 existence checks use `APICache` (SQLite, TTL in config). Repeat DOIs hit cache.
- L3/L5 in quick mode is near-instant (rule-based classification).
- L5 in agentic mode = triage + agent LLM calls (semaphore cap 5, see config).
- `paper2_attention.pdf`: skipped (GROBID not running at profile time).
- Cost extracted from `report.warnings`. Numbers appear materially lower than
  back-of-envelope token math suggests — cost accounting may be under-reporting
  (investigate in Phase 2 before setting real budget caps).

## Findings — where time actually goes

**Cold-cache baseline (P1-quick, 59 refs, 96s):**
- L2 existence dominates (92s / 96s = 96% of wall time). DOI + Crossref +
  OpenAlex + Semantic Scholar + PubMed cascade across 59 refs.

**Warm-cache APICache effect:**
- Same 59 refs, second run: L2 = 5s (was 92s). **~18× speedup from cache.**
- Once a paper is seen, re-running in a different mode is almost free on L2.

**Standard mode (P1-standard, 168s):**
- L4 comprehension = 163s. Passage retrieval + per-citation claim verification
  is the bottleneck, not L2 (warm here).

**Agentic mode (P1-agentic, 354s):**
- pre_retrieve: 153s (same passage retrieval as Standard L4 — duplicated work
  between modes; unifying these in Phase 2 is an easy win).
- triage: 1.9s (rule-based, negligible).
- metadata_dispatch: 14s (concurrency 5).
- claim_dispatch: 137s for 45 claim agents at concurrency 5
  (~15s per agent × 9 batches).
- both_dispatch: 42s (sequential metadata→claim per ref).

## Phase 2 optimization priorities (from this baseline)

1. **Raise `max_concurrent_agents` 5 → 15-20.** Claim dispatch is serialized
   into ~9 batches; raising concurrency cuts wall time near-linearly until
   OpenAI TPM/RPM rate limits bite.
2. **Share passage-retrieval output between standard and agentic modes.**
   Right now each mode redoes the ~150s retrieval pass independently.
3. **Cache full-text retrieval per DOI on disk** (not just APICache metadata).
   Fulltext fetches inside pre_retrieve / L4 are repeated across runs.
4. **Cold-cache L2 is the dominant cost for new papers.** Consider warming
   the cache from a shared corpus, or parallelizing the cross-database
   cascade further.
5. **Investigate cost under-reporting** before relying on `report.warnings`
   for budget enforcement in production.
