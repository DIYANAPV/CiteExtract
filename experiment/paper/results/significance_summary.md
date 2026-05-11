# Statistical significance summary

_Bootstrap CIs: 10,000 resamples, percentile method. McNemar's test: continuity-corrected chi-square (exact binomial when discordant pairs < 25). Holm-Bonferroni correction is applied within each comparison family (one family per natural rhetorical question). Stars: \* p<0.05, \*\* p<0.01, \*\*\* p<0.001, ns p>=0.05. Stars-Holm column applies the same thresholds to the Holm-adjusted p-value._


## Instance-alignment audit

All compared CSV pairs share identical instance ID sets — no instances dropped from any McNemar test.

## Metadata table — accuracy with 95% CI

| System | Acc % [95% CI] | n |
|---|---|---|
| Llama-3.1-8B | 45.03 [39.40, 50.66] | 302 |
| Qwen3-8B | 43.38 [38.08, 49.01] | 302 |
| GPT-4o-mini | 58.94 [53.64, 64.57] | 302 |
| GPT-4o-mini + search | 82.78 [78.48, 87.09] | 302 |
| GPT-4o | 61.59 [55.96, 66.89] | 302 |
| GPT-4o + search | 84.77 [80.79, 88.74] | 302 |
| GPT-5 (min) | 68.54 [63.25, 73.84] | 302 |
| GPT-5 + search (min) | 88.74 [85.10, 92.05] | 302 |
| GPT-5 (med) | 70.86 [65.56, 75.83] | 302 |
| GPT-5 + search (med) | 90.73 [87.42, 93.71] | 302 |
| GPT-5.5 (min) | 67.22 [61.92, 72.52] | 302 |
| GPT-5.5 + search (min) | 92.38 [89.40, 95.36] | 302 |
| GPT-5.5 (med) | 68.21 [62.91, 73.51] | 302 |
| GPT-5.5 + search (med) | 93.38 [90.40, 96.03] | 302 |
| CiteExtract + GPT-4o-mini | 96.36 [94.04, 98.34] | 302 |
| CiteExtract + GPT-5-mini | 97.02 [95.03, 98.68] | 302 |
| CiteExtract + GPT-5.5 (med) | 98.01 [96.36, 99.34] | 302 |

## Metadata pairwise McNemar

### metadata_vs_citeextract+gpt-4o-mini  (K=14 comparisons; Holm threshold for α=0.05)

| System | vs Reference | n_pair | Δ (Ref − Sys) | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | CiteExtract + GPT-4o-mini | 302 | +51.32 | 160 | 5 | 0 | 0 | *** | *** |
| Qwen3-8B | CiteExtract + GPT-4o-mini | 302 | +52.98 | 163 | 3 | 0 | 0 | *** | *** |
| GPT-4o-mini | CiteExtract + GPT-4o-mini | 302 | +37.42 | 118 | 5 | 0 | 0 | *** | *** |
| GPT-4o-mini + search | CiteExtract + GPT-4o-mini | 302 | +13.58 | 49 | 8 | 1.17e-07 | 7.02e-07 | *** | *** |
| GPT-4o | CiteExtract + GPT-4o-mini | 302 | +34.77 | 111 | 6 | 0 | 0 | *** | *** |
| GPT-4o + search | CiteExtract + GPT-4o-mini | 302 | +11.59 | 44 | 9 | 3.008e-06 | 1.504e-05 | *** | *** |
| GPT-5 (min) | CiteExtract + GPT-4o-mini | 302 | +27.81 | 91 | 7 | 0 | 0 | *** | *** |
| GPT-5 + search (min) | CiteExtract + GPT-4o-mini | 302 | +7.62 | 32 | 9 | 0.0005908 | 0.002363 | *** | ** |
| GPT-5 (med) | CiteExtract + GPT-4o-mini | 302 | +25.50 | 85 | 8 | 3.22e-15 | 2.254e-14 | *** | *** |
| GPT-5 + search (med) | CiteExtract + GPT-4o-mini | 302 | +5.63 | 26 | 9 | 0.006841 | 0.02052 | ** | * |
| GPT-5.5 (min) | CiteExtract + GPT-4o-mini | 302 | +29.14 | 98 | 10 | 1.11e-16 | 9.992e-16 | *** | *** |
| GPT-5.5 + search (min) | CiteExtract + GPT-4o-mini | 302 | +3.97 | 21 | 9 | 0.04461 | 0.08922 | * | ns |
| GPT-5.5 (med) | CiteExtract + GPT-4o-mini | 302 | +28.15 | 95 | 10 | 2.22e-16 | 1.776e-15 | *** | *** |
| GPT-5.5 + search (med) | CiteExtract + GPT-4o-mini | 302 | +2.98 | 18 | 9 | 0.1237 | 0.1237 | ns | ns |

### metadata_vs_citeextract+gpt-5.5(med)  (K=14 comparisons; Holm threshold for α=0.05)

| System | vs Reference | n_pair | Δ (Ref − Sys) | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | CiteExtract + GPT-5.5 (med) | 302 | +52.98 | 161 | 1 | 0 | 0 | *** | *** |
| Qwen3-8B | CiteExtract + GPT-5.5 (med) | 302 | +54.64 | 167 | 2 | 0 | 0 | *** | *** |
| GPT-4o-mini | CiteExtract + GPT-5.5 (med) | 302 | +39.07 | 120 | 2 | 0 | 0 | *** | *** |
| GPT-4o-mini + search | CiteExtract + GPT-5.5 (med) | 302 | +15.23 | 49 | 3 | 4.365e-10 | 2.619e-09 | *** | *** |
| GPT-4o | CiteExtract + GPT-5.5 (med) | 302 | +36.42 | 112 | 2 | 0 | 0 | *** | *** |
| GPT-4o + search | CiteExtract + GPT-5.5 (med) | 302 | +13.25 | 43 | 3 | 8.912e-09 | 4.456e-08 | *** | *** |
| GPT-5 (min) | CiteExtract + GPT-5.5 (med) | 302 | +29.47 | 92 | 3 | 0 | 0 | *** | *** |
| GPT-5 + search (min) | CiteExtract + GPT-5.5 (med) | 302 | +9.27 | 32 | 4 | 6.795e-06 | 2.718e-05 | *** | *** |
| GPT-5 (med) | CiteExtract + GPT-5.5 (med) | 302 | +27.15 | 85 | 3 | 0 | 0 | *** | *** |
| GPT-5 + search (med) | CiteExtract + GPT-5.5 (med) | 302 | +7.28 | 26 | 4 | 0.000126 | 0.0003781 | *** | *** |
| GPT-5.5 (min) | CiteExtract + GPT-5.5 (med) | 302 | +30.79 | 97 | 4 | 0 | 0 | *** | *** |
| GPT-5.5 + search (min) | CiteExtract + GPT-5.5 (med) | 302 | +5.63 | 21 | 4 | 0.001374 | 0.002749 | ** | ** |
| GPT-5.5 (med) | CiteExtract + GPT-5.5 (med) | 302 | +29.80 | 94 | 4 | 0 | 0 | *** | *** |
| GPT-5.5 + search (med) | CiteExtract + GPT-5.5 (med) | 302 | +4.64 | 18 | 4 | 0.004344 | 0.004344 | ** | ** |

### metadata_pipeline_pipeline  (K=3 comparisons; Holm threshold for α=0.05)

| System | vs Reference | n_pair | Δ (Ref − Sys) | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| CiteExtract + GPT-4o-mini | CiteExtract + GPT-5-mini | 302 | +0.66 | 6 | 4 | 0.7539 | 0.7539 | ns | ns |
| CiteExtract + GPT-4o-mini | CiteExtract + GPT-5.5 (med) | 302 | +1.66 | 6 | 1 | 0.125 | 0.375 | ns | ns |
| CiteExtract + GPT-5-mini | CiteExtract + GPT-5.5 (med) | 302 | +0.99 | 3 | 0 | 0.25 | 0.5 | ns | ns |

## Semantic table — accuracy with 95% CI

| Model × Condition | Acc % [95% CI] | n |
|---|---|---|
| Llama-3.1-8B × Title + Abstract | 52.36 [48.72, 56.01] | 741 |
| Llama-3.1-8B × Title + Abs + Passages | 53.98 [50.47, 57.62] | 741 |
| Llama-3.1-8B × Title only | 56.55 [53.04, 60.19] | 741 |
| Qwen3-8B × Title + Abstract | 71.79 [68.69, 74.90] | 741 |
| Qwen3-8B × Title + Abs + Passages | 83.00 [80.30, 85.56] | 741 |
| Qwen3-8B × Title only | 76.11 [73.14, 79.08] | 741 |
| GPT-4o-mini × Title + Abstract | 71.39 [68.15, 74.63] | 741 |
| GPT-4o-mini × Title + Abs + Passages | 83.81 [81.11, 86.50] | 741 |
| GPT-4o-mini × Title only | 72.74 [69.50, 75.98] | 741 |
| GPT-4o × Title + Abstract | 78.00 [75.03, 80.97] | 741 |
| GPT-4o × Title + Abs + Passages | 85.02 [82.46, 87.58] | 741 |
| GPT-4o × Title only | 80.43 [77.46, 83.27] | 741 |
| GPT-5 (med) × Title + Abstract | 72.60 [69.37, 75.84] | 741 |
| GPT-5 (med) × Title + Abs + Passages | 82.86 [80.16, 85.56] | 741 |
| GPT-5 (med) × Title only | 69.64 [66.40, 72.87] | 741 |
| GPT-5 (min) × Title + Abstract | 81.38 [78.54, 84.21] | 741 |
| GPT-5 (min) × Title + Abs + Passages | 86.37 [83.94, 88.80] | 741 |
| GPT-5 (min) × Title only | 81.38 [78.54, 84.08] | 741 |
| GPT-5.5 (med) × Title + Abstract | 73.01 [69.77, 76.25] | 741 |
| GPT-5.5 (med) × Title + Abs + Passages | 85.96 [83.54, 88.39] | 741 |
| GPT-5.5 (med) × Title only | 70.99 [67.75, 74.22] | 741 |
| GPT-5.5 (min) × Title + Abstract | 71.52 [68.29, 74.76] | 741 |
| GPT-5.5 (min) × Title + Abs + Passages | 86.10 [83.67, 88.53] | 741 |
| GPT-5.5 (min) × Title only | 69.23 [65.86, 72.47] | 741 |

## Semantic targeted pairs

### semantic_retrieval_effect  (K=8 comparisons)

| Comparison | n_pair | Acc A | Acc B | Δ | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B: title-only vs +passages | 741 | 56.55 | 53.98 | -2.56 | 93 | 112 | 0.2087 | 0.2087 | ns | ns |
| Qwen3-8B: title-only vs +passages | 741 | 76.11 | 83.00 | +6.88 | 103 | 52 | 5.917e-05 | 0.0002367 | *** | *** |
| GPT-4o-mini: title-only vs +passages | 741 | 72.74 | 83.81 | +11.07 | 137 | 55 | 5.045e-09 | 2.523e-08 | *** | *** |
| GPT-4o: title-only vs +passages | 741 | 80.43 | 85.02 | +4.59 | 68 | 34 | 0.001085 | 0.003255 | ** | ** |
| GPT-5 (min): title-only vs +passages | 741 | 81.38 | 86.37 | +4.99 | 82 | 45 | 0.001401 | 0.003255 | ** | ** |
| GPT-5 (med): title-only vs +passages | 741 | 69.64 | 82.86 | +13.23 | 142 | 44 | 1.141e-12 | 6.843e-12 | *** | *** |
| GPT-5.5 (min): title-only vs +passages | 741 | 69.23 | 86.10 | +16.87 | 157 | 32 | 0 | 0 | *** | *** |
| GPT-5.5 (med): title-only vs +passages | 741 | 70.99 | 85.96 | +14.98 | 143 | 32 | 1.11e-16 | 7.772e-16 | *** | *** |

### semantic_prod_vs_others_in_passages  (K=7 comparisons)

| Comparison | n_pair | Acc A | Acc B | Δ | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B vs GPT-4o-mini in +passages | 741 | 83.81 | 53.98 | -29.82 | 61 | 282 | 0 | 0 | *** | *** |
| Qwen3-8B vs GPT-4o-mini in +passages | 741 | 83.81 | 83.00 | -0.81 | 32 | 38 | 0.5501 | 1 | ns | ns |
| GPT-4o vs GPT-4o-mini in +passages | 741 | 83.81 | 85.02 | +1.21 | 36 | 27 | 0.3135 | 0.9405 | ns | ns |
| GPT-5 (min) vs GPT-4o-mini in +passages | 741 | 83.81 | 86.37 | +2.56 | 61 | 42 | 0.07613 | 0.4568 | ns | ns |
| GPT-5 (med) vs GPT-4o-mini in +passages | 741 | 83.81 | 82.86 | -0.94 | 56 | 63 | 0.5823 | 1 | ns | ns |
| GPT-5.5 (min) vs GPT-4o-mini in +passages | 741 | 83.81 | 86.10 | +2.29 | 64 | 47 | 0.1288 | 0.6442 | ns | ns |
| GPT-5.5 (med) vs GPT-4o-mini in +passages | 741 | 83.81 | 85.96 | +2.16 | 65 | 49 | 0.1601 | 0.6442 | ns | ns |

### semantic_narrative_pairs  (K=4 comparisons)

| Comparison | n_pair | Acc A | Acc B | Δ | b | c | p (raw) | p (Holm) | sig (raw) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o vs GPT-4o-mini in +passages (scale-with-retrieval) | 741 | 83.81 | 85.02 | +1.21 | 36 | 27 | 0.3135 | 0.8038 | ns | ns |
| GPT-5(min) vs GPT-4o in +passages (newer-frontier-doesnt-help) | 741 | 85.02 | 86.37 | +1.35 | 38 | 28 | 0.2679 | 0.8038 | ns | ns |
| GPT-5.5(min) vs GPT-5(min) in +passages | 741 | 86.37 | 86.10 | -0.27 | 37 | 39 | 0.9087 | 0.9087 | ns | ns |
| GPT-4o vs Qwen3-8B in +passages (open-weight-matches-frontier) | 741 | 83.00 | 85.02 | +2.02 | 53 | 38 | 0.1422 | 0.5689 | ns | ns |

## Suggested Evaluation Protocol paragraph (for the paper)

_For each pairwise comparison we report McNemar's test on per-instance correctness, with continuity correction for the chi-square approximation and an exact binomial test when fewer than 25 discordant pairs are observed. To control the family-wise error rate we apply the Holm-Bonferroni step-down correction within each comparison family (systems-vs-CiteExtract; pipeline-vs-pipeline; within-model retrieval effect; cross-model at fixed retrieval condition). We also report 95\% bootstrap confidence intervals on each accuracy estimate from 10{,}000 resamples of the test set with replacement._
