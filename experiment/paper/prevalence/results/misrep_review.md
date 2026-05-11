# Misrep review — 28 cases

Filter: verdict in ['CONTRADICTS', 'NOT_SUPPORTED'] AND paper_found AND full_text_available.

Each case below shows the citing sentence, the cited paper, the system's verdict + reasoning, and the passage it grounded the verdict on. Tick the appropriate box at the bottom of each case after reading the cited source.

---

## Case 1/28 — `0Nq1GvAGR0` / `30` — **CONTRADICTS**

- **Citing paper** [`0Nq1GvAGR0`](https://openreview.net/forum?id=0Nq1GvAGR0) (NeurIPS 2025 poster)
- **Cited:** Attention Based Vehicle Trajectory Prediction
- **DOI:** [10.1109/TIV.2020.2991952](https://doi.org/10.1109/TIV.2020.2991952)
- **Year:** 2020
- **Resolved by:** semantic_scholar · **Full text from:** oa_url

### Citing sentence
> (ii) knowledge-free methods, which rely on deep learning's ability to encode complex data features, modeling them using structures such as tensors [29,30] or attention mechanisms

### Surrounding context
**after:**  
> [31,32].

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper presents an attention-based trajectory prediction method and repeatedly emphasizes attention mechanisms and modeling vehicle interactions. It does not describe tensor-based modeling as the sentence attributes to it, so it is not an appropriate citation for 'modeling ... using structures such as tensors.'

**Evidence quote (passage the model used):**
> "Inspired by human reasoning, we use an attention mechanism that explicitly highlights the importance of neighboring vehicles with respect to their future states. We go beyond pairwise vehicle interactions and model higher order interactions. ... We exploit a combination of global and partial attention paid to surrounding vehicles to generate different possible trajectory."

**Top retrieved passage (raw):**
> Then, they jointly compute, for each pair, the probability of possible trajectories. Most recent studies deploy deep learning based methods. They will be detailed in the Section II-B. Planning-based methods reason on the motion intent of rational agents. Sierra González et al. [18] deploy Markov Decision Process (MDPs) to represent the driver decisionmaking strategy. They model a vehicle's trajectory by a sequence of states.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 2/28 — `59TL8MpBzX` / `7` — **CONTRADICTS**

- **Citing paper** [`59TL8MpBzX`](https://openreview.net/forum?id=59TL8MpBzX) (NeurIPS 2025 poster)
- **Cited:** MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction
- **DOI:** [10.1109/cvprw56347.2022.00090](https://doi.org/10.1109/cvprw56347.2022.00090)
- **Year:** 2022
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> Comparing Variant 1 with SimDiff-Spectra (i.e., a UNet-based HSI reconstruction network, which is similar to [7]), we find that the AICD primarily contributes to the spatial details recovery of the HSI signal, which aligns with our original design intention.

### Surrounding context
**before:**  
> Break-down Ablation. We perform bread-down ablation to investigate the effectiveness of each module in Table 3(a) and Figure 6.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> MST++ is presented as a Transformer-based framework (not a UNet/CNN-based network). Citing MST++ as an example of a UNet-based method is inconsistent with MST++'s description, so this citation is inappropriate.

**Evidence quote (passage the model used):**
> To address the aforementioned limitations, we propose the first Transformer-based framework, Multi-stage Spectral-wise Transformer (MST++) for efficient spectral reconstruction from RGB images.

**Top retrieved passage (raw):**
> In Sec. 3.4, we adopt three ensemble strategies for NTIRE 2022 Spectral Reconstruction Challenge. In this part, we perform ablations to study their effects. On the valid set, self-ensemble, multi-scale ensemble, and Top-K (K is set to 5) multi-model ensemble respectively achieve improvements by 0.015, 0.033, and 0.045 in terms of MRAE.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 3/28 — `5xwyxupsLL` / `31` — **CONTRADICTS**

- **Citing paper** [`5xwyxupsLL`](https://openreview.net/forum?id=5xwyxupsLL) (NeurIPS 2025 poster)
- **Cited:** One-Step Diffusion with Distribution Matching Distillation
- **DOI:** [10.1109/cvpr52733.2024.00632](https://doi.org/10.1109/cvpr52733.2024.00632)
- **Year:** 2024
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> For models with only a handful of steps-such as distilled variants like the 4-step Flux.1-schnell [31] or one-step diffusion methods [32,33]-temporal redundancy is minimal, making PipeFusion less effective.

### Surrounding context
**before:**  
> A key limitation of PipeFusion lies in its dependence on multiple diffusion steps.
**after:**  
> However, this does not pose a major constraint in practice, as state-of-the-art DiT models generally require a substantial number of steps to deliver competitive quality.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (One-Step Diffusion with Distribution Matching Distillation) presents a distillation pipeline that reduces the multi-step generative process to a single forward pass (i.e., one-step diffusion), not a 4-step distilled variant. Thus it does not support the specific claim that it is an example of a "4-step Flux.1-schnell" model; the citation is misattributed. While the paper is relevant as an example of small-step/one-step distilled models (which is topically related), it contradicts the specific labeling as a 4-step variant.

**Evidence quote (passage the model used):**
> Our work builds upon leading diffusion models [31,63] and introduces a simple distillation pipeline that reduces the multi-step generative process to a single forward pass. Our method is universally applicable to any diffusion model with deterministic sampling [31,72,74].

**Top retrieved passage (raw):**
> Diffusion Acceleration Accelerating the inference process of diffusion models has been a key focus in the field, leading to the development of two types of approaches. The first type advances fast diffusion samplers [31,41,45,46,91], which can dramatically reduce the number of sampling steps required by pre-trained diffusion models-from a thousand down to merely 20-50. However, a further reduction in steps often results in a catastrophic decrease in performance.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 4/28 — `AmZ7uHDJiR` / `7` — **CONTRADICTS**

- **Citing paper** [`AmZ7uHDJiR`](https://openreview.net/forum?id=AmZ7uHDJiR) (NeurIPS 2025 poster)
- **Cited:** Neural Topological SLAM for Visual Navigation
- **DOI:** [10.1109/cvpr42600.2020.01289](https://doi.org/10.1109/cvpr42600.2020.01289)
- **Year:** 2020
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> Most SLAM systems [38,34,7,58,60,56,49,20,27,16] assume an autonomous agent navigating an environment with distant, static illumination, e.g., a self-driving car in the streets, and they optimize a Photometric Bundle Adjustment loss where they minimize an error between the captured image and the re-rendered image using estimated 3D scene and camera poses.

### Surrounding context
**before:**  
> Simultaneous Localization and Mapping (SLAM) enables autonomous agents to build a spatial map of an unknown environment while estimating their own poses within it, with wide-ranging applications in robotics, computer vision, autonomous vehicles, and scientific imaging.
**after:**  
> Photo Photo Static Lighting Dynamic Lighting Figure 2: MonoGS performance under (1) distant static lighting and (2) dynamic near-field lighting from a co-located flashlight.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (Neural Topological SLAM, NTS) describes a topological SLAM approach that builds a graph of panoramic images and relative poses for navigation, not a metric photometric bundle-adjustment system that minimizes image vs. re-rendered image photometric error. Thus it does not support the claim that 'most SLAM systems' optimize a photometric BA loss; in fact NTS exemplifies a different (topological) SLAM paradigm and is not evidence for the photometric BA assertion.

**Evidence quote (passage the model used):**
> We propose a modular model, 'Neural Topological SLAM (NTS)', which builds and maintains a topological map for navigation. The topological map is represented using a graph, denoted by G_t at time t. Each node in the graph (N_i) is associated with a panoramic image (I_{N_i}) and represents the area visible in this image. Two nodes are connected by an edge (E_{i,j}) if they represent adjacent areas. Each edge also stores the relative pose between two nodes, ∆p_{ij}.

**Top retrieved passage (raw):**
> We use the Localization model and Local Policy from NTS to detect when a goal is nearby and navigate to it. Active Neural SLAM. This is a recent modular model based on Metric Spatial Maps proposed for the task of exploration. We adapt it to the Image Goal task by using the Localization model and Local Policy from NTS to detect when a goal is nearby and navigate to it.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 5/28 — `AuOZDp4gy7` / `36` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** TS2Vec: Towards Universal Representation of Time Series
- **DOI:** [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- **Year:** 2022
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> The best results are highlighted in bold.0.625±0.001.391±0.001 0.044±0.002 0.429±0.001 2.010±0.003
> Table 18 :18Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The maximum improvement of AliO over the baseline is MSE: 36.07%, TAM: 77.73%, MAE: 8.21%, MAPE: 15.19%, and RMSE: 5.61%. ECL, ETTh1, ETTh2, ETTm1, and ETTm2 are in Tab. 14.Models
> Table 17 :17Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and PatchTST[33].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The sentence uses [36] to refer to TimesNet. The cited paper (36) is TS2Vec, not TimesNet, so the citation does not support the claim and appears to be incorrect.

**Evidence quote (passage the model used):**
> The evaluation results are summarized in Table 1. TS2Vec achieves substantial improvement compared to other representation learning methods on both UCR and UEA datasets.

**Top retrieved passage (raw):**
> Table 4 shows the performance comparison of different methods on F 1 score, precision and recall. In the normal setting, TS2Vec improves the F 1 score by 18.2% on Yahoo dataset and 5.5% on KPI dataset compared to the best result of baseline methods. In the cold-start setting, the F 1 score is improved by 19.7% on Yahoo dataset and 1.0% on KPI dataset than the best SOTA result. Note that our method achieves similar scores on these two settings, demonstrating the transferability of TS2Vec from one dataset to another.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 6/28 — `AuOZDp4gy7` / `36` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** TS2Vec: Towards Universal Representation of Time Series
- **DOI:** [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- **Year:** 2022
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> The best results are highlighted in bold.0.625±0.001.391±0.001 0.044±0.002 0.429±0.001 2.010±0.003
> Table 18 :18Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The maximum improvement of AliO over the baseline is MSE: 36.07%, TAM: 77.73%, MAE: 8.21%, MAPE: 15.19%, and RMSE: 5.61%. ECL, ETTh1, ETTh2, ETTm1, and ETTm2 are in Tab. 14.Models
> Table 17 :17Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and PatchTST[33].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The sentence uses [36] to refer to TimesNet. The cited paper (36) is TS2Vec, not TimesNet, so the citation does not support the claim and appears to be incorrect.

**Evidence quote (passage the model used):**
> The evaluation results are summarized in Table 1. TS2Vec achieves substantial improvement compared to other representation learning methods on both UCR and UEA datasets.

**Top retrieved passage (raw):**
> Table 4 shows the performance comparison of different methods on F 1 score, precision and recall. In the normal setting, TS2Vec improves the F 1 score by 18.2% on Yahoo dataset and 5.5% on KPI dataset compared to the best result of baseline methods. In the cold-start setting, the F 1 score is improved by 19.7% on Yahoo dataset and 1.0% on KPI dataset than the best SOTA result. Note that our method achieves similar scores on these two settings, demonstrating the transferability of TS2Vec from one dataset to another.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 7/28 — `AuOZDp4gy7` / `36` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** TS2Vec: Towards Universal Representation of Time Series
- **DOI:** [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- **Year:** 2022
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> 19.Models
> Table 19 :19Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The maximum improvement of AliO over the baseline is MSE: 29.03%, TAM: 51.96%, MAE: 9.34%, MAPE: 23.09%, and RMSE: 16.56%. Exchange, Traffic, ILI, and Weather are in Tab.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> This repeats the same misattribution: the text cites [36] for TimesNet, but [36] is TS2Vec. TS2Vec does not verify or represent TimesNet, so the citation is incorrect for that claim.

**Evidence quote (passage the model used):**
> Table 4 shows the performance comparison of different methods on F1 score, precision and recall. In the normal setting, TS2Vec improves the F1 score by 18.2% on Yahoo dataset and 5.5% on KPI dataset compared to the best result of baseline methods.

**Top retrieved passage (raw):**
> Table 4 shows the performance comparison of different methods on F 1 score, precision and recall. In the normal setting, TS2Vec improves the F 1 score by 18.2% on Yahoo dataset and 5.5% on KPI dataset compared to the best result of baseline methods. In the cold-start setting, the F 1 score is improved by 19.7% on Yahoo dataset and 1.0% on KPI dataset than the best SOTA result. Note that our method achieves similar scores on these two settings, demonstrating the transferability of TS2Vec from one dataset to another.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 8/28 — `AuOZDp4gy7` / `36` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** TS2Vec: Towards Universal Representation of Time Series
- **DOI:** [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- **Year:** 2022
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> 19.Models
> Table 19 :19Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The maximum improvement of AliO over the baseline is MSE: 29.03%, TAM: 51.96%, MAE: 9.34%, MAPE: 23.09%, and RMSE: 16.56%. Exchange, Traffic, ILI, and Weather are in Tab.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> This repeats the same misattribution: the text cites [36] for TimesNet, but [36] is TS2Vec. TS2Vec does not verify or represent TimesNet, so the citation is incorrect for that claim.

**Evidence quote (passage the model used):**
> Table 4 shows the performance comparison of different methods on F1 score, precision and recall. In the normal setting, TS2Vec improves the F1 score by 18.2% on Yahoo dataset and 5.5% on KPI dataset compared to the best result of baseline methods.

**Top retrieved passage (raw):**
> Table 4 shows the performance comparison of different methods on F 1 score, precision and recall. In the normal setting, TS2Vec improves the F 1 score by 18.2% on Yahoo dataset and 5.5% on KPI dataset compared to the best result of baseline methods. In the cold-start setting, the F 1 score is improved by 19.7% on Yahoo dataset and 1.0% on KPI dataset than the best SOTA result. Note that our method achieves similar scores on these two settings, demonstrating the transferability of TS2Vec from one dataset to another.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 9/28 — `AuOZDp4gy7` / `9` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- **DOI:** [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- **Year:** 2023
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> We
> Table 23 :23Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE. The best results are highlighted in bold.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> Duplicate of the previous table citation: the paper being cited is about self-supervised contrastive methods for semi-supervised classification, not GPT4TS. The attribution therefore appears incorrect.

**Evidence quote (passage the model used):**
> Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification

**Top retrieved passage (raw):**
> It is clear that using a combination of weak and strong augmentations yields the best performance. The results are obtained with the linear evaluation experiment on 5% of labeled data on three datasets.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 10/28 — `AuOZDp4gy7` / `9` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- **DOI:** [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- **Year:** 2023
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> We
> Table 23 :23Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE. The best results are highlighted in bold.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> Duplicate of the previous table citation: the paper being cited is about self-supervised contrastive methods for semi-supervised classification, not GPT4TS. The attribution therefore appears incorrect.

**Evidence quote (passage the model used):**
> Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification

**Top retrieved passage (raw):**
> It is clear that using a combination of weak and strong augmentations yields the best performance. The results are obtained with the linear evaluation experiment on 5% of labeled data on three datasets.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 11/28 — `AuOZDp4gy7` / `9` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- **DOI:** [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- **Year:** 2023
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> 24.We use official GitHub code (https:
> Table 24 :24Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The maximum improvement of AliO over the baseline is MSE: 11.54%, TAM: 62.06%, MAE: 10.98%, MAPE: 8.16%, and RMSE: 5.93%. ILI, Traffix, and Weather are in Tab.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> Same as above: the cited paper's topic (self-supervised contrastive learning for semi-supervised classification) does not match the claimed model name GPT4TS used in the table, indicating the citation is incorrect for this claim.

**Evidence quote (passage the model used):**
> Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification

**Top retrieved passage (raw):**
> Table 2 shows the linear evaluation results of our approaches against the baseline methods using 1% and 5% of the labeled data. The results show that TS-TCC achieves the best overall performance across all 10 datasets in both the 1% and 5% label scenarios. Specifically, TS-TCC is the best performer on six datasets and the second best on two datasets with only a small margin behind the best methods.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 12/28 — `AuOZDp4gy7` / `9` — **CONTRADICTS**

- **Citing paper** [`AuOZDp4gy7`](https://openreview.net/forum?id=AuOZDp4gy7) (NeurIPS 2025 poster)
- **Cited:** Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- **DOI:** [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- **Year:** 2023
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> 24.We use official GitHub code (https:
> Table 24 :24Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Surrounding context
**before:**  
> The maximum improvement of AliO over the baseline is MSE: 11.54%, TAM: 62.06%, MAE: 10.98%, MAPE: 8.16%, and RMSE: 5.93%. ILI, Traffix, and Weather are in Tab.
**after:**  
> The best results are highlighted in bold.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> Same as above: the cited paper's topic (self-supervised contrastive learning for semi-supervised classification) does not match the claimed model name GPT4TS used in the table, indicating the citation is incorrect for this claim.

**Evidence quote (passage the model used):**
> Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification

**Top retrieved passage (raw):**
> Table 2 shows the linear evaluation results of our approaches against the baseline methods using 1% and 5% of the labeled data. The results show that TS-TCC achieves the best overall performance across all 10 datasets in both the 1% and 5% label scenarios. Specifically, TS-TCC is the best performer on six datasets and the second best on two datasets with only a small margin behind the best methods.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 13/28 — `T4qJuQCFAK` / `10` — **CONTRADICTS**

- **Citing paper** [`T4qJuQCFAK`](https://openreview.net/forum?id=T4qJuQCFAK) (NeurIPS 2025 poster)
- **Cited:** Orthogonal Gradient Descent for Continual Learning
- **Year:** 2019
- **Resolved by:** semantic_scholar · **Full text from:** arxiv

### Citing sentence
> Notably, the saliency of weights is determined by analyzing the activation space (sensitivities) of the base model rather than just regularizing based on the weight space, similar to [10,26].

### Surrounding context
**before:**  
> In this way, AIM is analogous to weight regularization in many continual learning approaches [1,22,31,33,39].
**after:**  
> Activation Space Analysis

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> OGD is described as a parameter-space/gradient-projection method that restricts gradient directions to avoid interference with previous tasks; it does not present an approach that determines weight saliency via activation-space sensitivity analysis. Citing OGD as similar to an activation-space saliency method is therefore misleading.

**Evidence quote (passage the model used):**
> we propose to address this issue from a parameter space perspective and study an approach to restrict the direction of the gradient updates to avoid forgetting previously-learned data. We present the Orthogonal Gradient Descent (OGD) method, which accomplishes this goal by projecting the gradients from new tasks onto a subspace in which the neural network output on previous task does not change...

**Top retrieved passage (raw):**
> We performed experiments in this section on three continual learning benchmark: Permuted Mnist, Rotated Mnist, and Split Mnist. Baselines. We considered the following baselines for comparison purposes. (1) EWC (Kirkpatrick et al., 2017): one of the pioneering regularization based methods that uses fisher information diagonals as important weights. (2) A-GEM (Chaudhry et al., 2018): using loss gradients of stored previous data in an inequality constrained optimization. (3) SGD: Stochastic Gradient Descent optimizing tasks one after the other.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 14/28 — `T4qJuQCFAK` / `10` — **CONTRADICTS**

- **Citing paper** [`T4qJuQCFAK`](https://openreview.net/forum?id=T4qJuQCFAK) (NeurIPS 2025 poster)
- **Cited:** Orthogonal Gradient Descent for Continual Learning
- **Year:** 2019
- **Resolved by:** semantic_scholar · **Full text from:** arxiv

### Citing sentence
> An alternative way of choosing the importance scores for changing the model weights-which has been used, e.g., in continual learning [10,26], out-of-distribution detection

### Surrounding context
**before:**  
> Sensitivity-Based Formulation
**after:**  
> [32], and meta-learning

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The sentence implies that [10] (OGD) is an example of a sensitivity-based method that chooses importance scores for weights. OGD instead operates by projecting gradients in parameter space to avoid forgetting, not by computing sensitivity-based importance scores for weights, so this specific attribution is incorrect.

**Evidence quote (passage the model used):**
> In this work, we apply the proposed Orthogonal Gradient Descent (OGD) algorithm to continual learning on consecutive tasks. ... projecting the gradients from new tasks onto a subspace in which the neural network output on previous task does not change and the projected gradient is still in a useful direction for learning the new task.

**Top retrieved passage (raw):**
> In the regularization based approaches, catastrophic forgetting is tackled by imposing constraints on the weight updates of the neural network according to some importance measure for previous tasks. The difference lies in the way how importance weights are computed.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 15/28 — `fru52tkjHf` / `8` — **CONTRADICTS**

- **Citing paper** [`fru52tkjHf`](https://openreview.net/forum?id=fru52tkjHf) (NeurIPS 2025 poster)
- **Cited:** signSGD: compressed optimisation for non-convex problems
- **Year:** 2018
- **Resolved by:** semantic_scholar · **Full text from:** arxiv

### Citing sentence
> [Bernstein et al., 2018], may fail to converge [Karimireddy et al., 2019].

### Surrounding context
**before:**  
> It is an interesting future topic to prove this conjecture and further explore the nonconvex behavior of ASGO theoretically. Also, we note that as an intuitively smoother version of Muon, ASGO has good convergence properties in nonsmooth settings as shown in Section 4, where Muon, like SignSGD
**after:**  
> Remark 4.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The signSGD paper presents convergence results and convergence-rate theorems for SIGNSGD/SIGNUM under its assumptions and argues SIGNSGD achieves SGD-level convergence; it does not state that SignSGD may fail to converge in the general sense. Using Bernstein et al. to support the claim that SignSGD "may fail to converge" is therefore not supported by the cited paper (and in fact the paper offers positive convergence results), so the citation is inappropriate for that negative claim.

**Evidence quote (passage the model used):**
> Training large neural networks requires distributing learning across multiple workers... We prove that it can get the best of both worlds: compressed gradients and SGD-level convergence rate. ... Using a theorem by Gauss, we prove that the non-convex convergence rate of majority vote matches that of distributed SGD.

**Top retrieved passage (raw):**
> Our work touches upon interesting aspects of the geometry of high-dimensional error surfaces, which we wish to explore in future work. But the next step for us will be to reach out to members of the distributed systems community to help benchmark the majority vote algorithm which shows such great theoretical promise for 1-bit compression in both directions between parameter-server and workers.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 16/28 — `fru52tkjHf` / `24` — **CONTRADICTS**

- **Citing paper** [`fru52tkjHf`](https://openreview.net/forum?id=fru52tkjHf) (NeurIPS 2025 poster)
- **Cited:** AURA: An Adaptive Unified Regularization Approach for Gradient-Based Optimization
- **DOI:** [10.21203/rs.3.rs-7480833/v1](https://doi.org/10.21203/rs.3.rs-7480833/v1)
- **Year:** 2025
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> [Gupta et al., 2017] and mainly highlight its superior convergence results compared to Shampoo and fullmatrix AdaGrad.

### Surrounding context
**before:**  
> features.10 features.12 features.14 features.17 features.19 features.21 features.24 features.26 features.28 classifier.0 classifier.3 classifier.6 features.0 features.2 features.5 features.7 features.10 features.12 features.14 features.17 features.19 features.21 features.24 features.26 features.28 classifier.0 classifier.3 classifier.6 0.0 0.5 1.0 1.5 2.0 2.5 3.0 (a) VGG16 patch_embed.proj blocks.0.attn.qkv blocks.0.attn.proj blocks.0.mlp.fc1 blocks.0.mlp.fc2 blocks.6.attn.qkv blocks.6.attn.proj blocks.6.mlp.fc1 blocks.6.mlp.fc2 blocks.11.attn.qkv blocks.11.attn.proj blocks.11.mlp.fc1 blocks.11.mlp.fc2 head patch_embed.proj blocks.0.attn.qkv blocks.0.attn.proj blocks.0.mlp.fc1 blocks.0.mlp.fc2 blocks.6.attn.qkv blocks.6.attn.proj blocks.6.mlp.fc1 blocks.6.mlp.fc2 blocks.11.attn.qkv blocks.11.attn.proj blocks.11.mlp.fc1 blocks.11.mlp.fc2 head 0.0 0.5 1.0 1.5 2.0 2.5 3.0 (b) ViT-base wte h.0.attn.wq h.0.attn.wk h.0.attn.wv h.0.attn.wo h.0.mlp.c_fc h.0.mlp.c_proj h.3.attn.wq h.3.attn.wk h.3.attn.wv h.3.attn.wo h.3.mlp.c_fc h.3.mlp.c_proj h.7.attn.wq h.7.attn.wk h.7.attn.wv h.7.attn.wo h.7.mlp.c_fc h.7.mlp.c_proj h.11.attn.wq h.11.attn.wk h.11.attn.wv h.11.attn.wo h.11.mlp.c_fc h.11.mlp.c_proj wte h.0.attn.wq h.0.attn.wk h.0.attn.wv h.0.attn.wo h.0.mlp.c_fc h.0.mlp.c_proj h.3.attn.wq h.3.attn.wk h.3.attn.wv h.3.attn.wo h.3.mlp.c_fc h.3.mlp.c_proj h.7.attn.wq h.7.attn.wk h.7.attn.wv h.7.attn.wo h.7.mlp.c_fc h.7.mlp.c_proj h.11.attn.wq h.11.attn.wk h.11.attn.wv h.11.attn.wo h.11.mlp.c_fc h.11. authors develop their method from the unified preconditioning method framework AdaptReg
**after:**  
> In our paper, we focus more on theoretically discussing how ASGO can utilize the structured properties of optimization problems including low-rank gradients and block-wise diagonal Hessians, to highlight the potential of ASGO as a practical algorithm for training deep learning models.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (AURA) presents a new optimizer called AURA and compares it experimentally to Adam, AdamW, AMSGrad, and RMSProp. It does not mention the AdaptReg unified preconditioning framework, nor does it report comparisons against Shampoo or full-matrix AdaGrad. Thus it does not support the claim that the authors developed their method from AdaptReg or that the paper 'mainly highlight[s]' superiority over Shampoo/full-matrix AdaGrad — the attribution is incorrect.

**Evidence quote (passage the model used):**
> In this work, we introduced AURA (Adaptive Unified Regularized Algorithm), a novel gradientbased optimization algorithm that unifies multiple adaptive mechanisms into a single framework... Through experiments on both regression (California Housing dataset) and classification tasks (Breast Cancer, and synthetic Make Moons), we demonstrated that AURA achieves comparable or superior generalization performance relative to widely used optimizers such as Adam, AdamW, AMSGrad and RMSProp.

**Top retrieved passage (raw):**
> In this work, we introduced AURA (Adaptive Unified Regularized Algorithm), a novel gradientbased optimization algorithm that unifies multiple adaptive mechanisms into a single framework. Unlike traditional optimizers such as Adam or RMSProp, AURA incorporates loss trend dynamics, gradient norm scaling, and trajectory similarity via cosine modulation to adapt the momentum coefficient β t in a principled manner. This approach enables the optimizer to retain stability during volatile training phases while accelerating convergence when gradients are consistent.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 17/28 — `g42mGfR6We` / `40` — **CONTRADICTS**

- **Citing paper** [`g42mGfR6We`](https://openreview.net/forum?id=g42mGfR6We) (NeurIPS 2025 poster)
- **Cited:** Prompt Guided Transformer for Multi-Task Dense Prediction
- **DOI:** [10.1109/tmm.2024.3349865](https://doi.org/10.1109/tmm.2024.3349865)
- **Year:** 2024
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> Focusing on this problem, many adapter-based methods have been proposed [37,50,57,65,40].

### Surrounding context
**before:**  
> Multi-task adaptation aims to fine-tune a pre-trained foundation model simultaneously or sequentially in multiple downstream tasks [34,32].
**after:**  
> In particular, denote the data of K tasks as {D k } K k=1 .

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (PGT) presents a prompt-guided, task-conditional transformer that uses prompting / learnable task prompts as a param-efficient adaptation method, not traditional adapter modules. The paper frames its approach as prompting / task-conditioned prompts rather than inserting adapter layers, so treating it as an "adapter-based" method is a misattribution.

**Evidence quote (passage the model used):**
> "Our PGT differs from these methods in that we design a transformer-based model with a lightweight task-conditional framework, providing a promising solution to address the issue of high parameter counts. d) Prompting: Prompting is a technique that inserts language instructions into the input space ... adapt pre-trained models to downstream tasks by only fine-tuning a small portion of parameters [23]."

**Top retrieved passage (raw):**
> Our PGT differs from these methods in that we design a transformer-based model with a lightweight task-conditional framework, providing a promising solution to address the issue of high parameter counts. d) Prompting: Prompting is a technique that inserts language instructions into the input space to help a transformerbased language model understand a task and adapt pre-trained models to downstream tasks by only fine-tuning a small portion of parameters [23].

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 18/28 — `kND7h1kD53` / `26` — **CONTRADICTS**

- **Citing paper** [`kND7h1kD53`](https://openreview.net/forum?id=kND7h1kD53) (NeurIPS 2025 poster)
- **Cited:** A Euclidean transformer for fast and stable machine learned force fields
- **DOI:** [10.1038/s41467-024-50620-6](https://doi.org/10.1038/s41467-024-50620-6)
- **Year:** 2024
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> Unlike existing methods which either use nonlocal attention using spherical harmonics [26] in Euclidean-space neighborhoods or using high-degree steerable representations within a single geometric space [17,6], DualEquiNet jointly constructs complementary representations in both Euclidean (EU) and spherical harmonics (SH) spaces: the EU space captures local geometric interactions crucial for atomiclevel detail, while the SH space encodes rotation-equivariant features well-suited for long-range structural dependencies.

### Surrounding context
**before:**  
> To address the challenges of hierarchical structure and long-range dependencies in large biomolecules, we propose DualEquiNet, a novel Dual-Space Hierarchical Equivariant Network.
**after:**  
> Through bidirectional cross-space message passing, DualEquiNet enables structured information exchange between EU and SH spaces, integrating local and global geometric contexts.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The citing sentence attributes to [26] the use of "nonlocal attention using spherical harmonics in Euclidean-space neighborhoods." The cited paper instead emphasizes a Euclidean self-attention mechanism that replaces SO(3) convolutions (i.e., it is proposing an alternative to spherical-harmonic / SO(3)-convolution approaches), so it does not support the described category. The paper presents Euclidean self-attention / SO3KRATES rather than a spherical-harmonics nonlocal-attention method, making the citation inappropriate for that specific claim.

**Evidence quote (passage the model used):**
> We take this as motivation to propose a Euclidean self-attention mechanism that replaces SO(3) convolutions with a filter on the relative orientation of atomic neighborhoods, representing atomic interactions without the need for expensive tensor products.

**Top retrieved passage (raw):**
> We overcome the challenging trade-off between stability and computational cost by combining two concepts-a Euclidean selfattention mechanism and the EV as efficient representation for molecular geometry-within the equivariant transformer architecture SO3KRATES. The exceptional performance of our approach is due to the decoupling of invariant and equivariant information, which enables a substantial reduction in computational complexity compared to other equivariant models.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 19/28 — `kND7h1kD53` / `65` — **CONTRADICTS**

- **Citing paper** [`kND7h1kD53`](https://openreview.net/forum?id=kND7h1kD53) (NeurIPS 2025 poster)
- **Cited:** E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction
- **Year:** 2022
- **Resolved by:** openreview · **Full text from:** oa_url

### Citing sentence
> 3D RNA structures are generated using RhoFold [65] following prior works [57].

### Surrounding context
**before:**  
> [74], which provides nucleotide-level reactivity and degradation labels relevant for vaccine stability; (ii) Ribonanza [37], which includes nucleotide-level reactivity under different compounds to assess RNA folding and flexibility; and (iii) Tc-ribo [33], which contains RNA-level regulatory behavior in response to tetracycline.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (reference [65]) is E2Efold-3D, which introduces a different method named E2Efold-3D for de novo RNA 3D structure prediction. It does not present or document a tool called RhoFold, so citing it as the source for RhoFold is incorrect.

**Evidence quote (passage the model used):**
> Title and abstract: "E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction" / "In this work, we develop the first end-to-end deep learning approach, E2Efold-3D, to accurately perform the de novo RNA structure prediction."

**Top retrieved passage (raw):**
> Figure 3(b) is a structure of Thg1-like protein (TLP) with tRNA (Phe), which is relatively long with 251 nucleotides. We have obtained surprisingly accurate results, with an RMSD of 1.29 Å compared to 18.06 Å of FARFAR2. The length of the nucleotide chain does not adversely affect our performance in this case. And we achieve a significantly higher TM-score (0.8445) than FARFAR2 (0.2345), in contrast to (a). Thus, the above example demonstrates the efficacy of the proposed approach.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 20/28 — `kND7h1kD53` / `79` — **CONTRADICTS**

- **Citing paper** [`kND7h1kD53`](https://openreview.net/forum?id=kND7h1kD53) (NeurIPS 2025 poster)
- **Cited:** Shape-aware Graph Spectral Learning
- **DOI:** [10.1145/3627673.3679604](https://doi.org/10.1145/3627673.3679604)
- **Year:** 2024
- **Resolved by:** openalex · **Full text from:** s2_fallback

### Citing sentence
> A variety of benchmarks [24,70,86,79] have been proposed to study this problem, along with several representative solutions, including Implicit Graph Neural Networks (IGNNs) and Graph Transformers (GTs).

### Surrounding context
**before:**  
> Learning long-range dependencies is a key challenge in graph machine learning, arising from the fundamental trade-off between under-reaching (i.e., insufficient GNN layers leading to small receptive fields) and over-smoothing (i.e., excessive layers causing feature homogenization) [18,58].
**after:**  
> IGNNs define message-passing through fixed-point equations [34,49] or unfolded optimization steps [20,8], allowing stable convergence to an equilibrium state even with infinite layers.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (Shape-aware Graph Spectral Learning) is a methods paper that analyzes spectral filters and proposes a regularization and a model (NewtonNet), with experiments on existing datasets. It does not present itself as a benchmark or a benchmark suite proposed to study long-range dependency problems. Thus it is not an appropriate citation for the claim that it is a proposed benchmark.

**Evidence quote (passage the model used):**
> "To mitigate this gap, we conduct theoretical and empirical analyses revealing a positive correlation between low-frequency importance and the homophily ratio ... Motivated by this, we propose shape-aware regularization on a Newton Interpolation-based spectral filter ... Comprehensive experiments demonstrate that NewtonNet can achieve graph spectral filters with desired shapes and superior performance on both homophilous and heterophilous datasets."

**Top retrieved passage (raw):**
> In real-world scenarios, different graphs can have different homophily ratios. In homophilous graphs, nodes from the same class tend to be connected; while in heterophilous graphs, nodes from different classes tend to be connected. Several studies (Zhu et al., 2020a;Ma et al., 2022;Xu et al., 2022) have shown that heterophilous graphs pose a challenge for spatial GNNs based on message-passing as they implicitly assume the graph follows homophily assumption.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 21/28 — `kND7h1kD53` / `65` — **CONTRADICTS**

- **Citing paper** [`kND7h1kD53`](https://openreview.net/forum?id=kND7h1kD53) (NeurIPS 2025 poster)
- **Cited:** E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction
- **Year:** 2022
- **Resolved by:** openreview · **Full text from:** oa_url

### Citing sentence
> In our experiments, we infer the 3D atomic structure for each sequence using RhoFold [65].

### Surrounding context
**before:**  
> Sequences with low signal-to-noise ratio (SNR < 1) are filtered out to ensure label reliability, following prior best practices. This dataset is especially suited for testing fine-grained, residue-level predictions.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The paper cited as [65] presents E2Efold-3D (an end-to-end deep learning method). It does not appear to be RhoFold, so it cannot legitimately be cited as the source or implementation of RhoFold used in the experiments.

**Evidence quote (passage the model used):**
> Abstract: "In this work, we develop the first end-to-end deep learning approach, E2Efold-3D, to accurately perform the de novo RNA structure prediction."

**Top retrieved passage (raw):**
> RNAs that exceed the range of 20-512 nucleotides. Eventually, we construct a training dataset with 1002 structures, a testing dataset with 50 structures, and an independent testing dataset with 30 challenging RNA targets from the community-wide RNA-Puzzles experiments. The raw RNA-Puzzle dataset is publicly available at Github. As the training dataset size is very small, we attempt to augment the data. To begin with, we re-integrating the sequence clusters which have been redundancy-filtered back into the training data.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 22/28 — `kND7h1kD53` / `74` — **CONTRADICTS**

- **Citing paper** [`kND7h1kD53`](https://openreview.net/forum?id=kND7h1kD53) (NeurIPS 2025 poster)
- **Cited:** Deep learning models for predicting RNA degradation via dual crowdsourcing
- **DOI:** [10.1038/s42256-022-00571-8](https://doi.org/10.1038/s42256-022-00571-8)
- **Year:** 2022
- **Resolved by:** openalex · **Full text from:** html_meta_pdf

### Citing sentence
> ModelTrain Time (s/epoch) Memory (MB)SchNet4.361628EGNN6.240608GVPGNN11.17210160TFN56.23543970FastEGNN11.7551954HEGNN6.3201232GotenNet10.3325894DualEqui10.9902436D.6 Datasets DetailsCOVID Vaccine The OpenVaccine COVID-19 dataset [74] consists of 4,082 synthetic RNAsequences, each 107-130 nucleotides long.

### Surrounding context
**before:**  
> At each layer, EU and SH neighborhoods are dynamically redefined based on updated 3D coordinates and SH features, respectively. We compute 4 statistics:
> Table 9 :9Training time and memory usage of each model.
**after:**  
> Table 10 :10Dataset Statistics.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper reports a different dataset size and length range: the OpenVaccine Kaggle dataset comprises 6,043 RNA constructs of 102–130 nucleotides (not 4,082 sequences of 107–130 nt). The available abstract and text thus contradict the specific numeric claim in the citing sentence.

**Evidence quote (passage the model used):**
> single-nucleotide resolution measurements on 6,043 diverse 102-130-nucleotide RNA constructs

**Top retrieved passage (raw):**
> Average signal/noise ratio for SHAPE reactivity across each dataset increased from 5.3±2.4 (mean ± standard deviation) to 6.2±3.5; for deg_Mg_pH10, 4.1±2.0 to 6.4±3.8 for Rounds 1 and 2; and for deg_Mg_50C 3.87±1.8 to 5.3±3.1 (Extended Data Fig. 5).

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 23/28 — `qMm7tC1zvj` / `58` — **CONTRADICTS**

- **Citing paper** [`qMm7tC1zvj`](https://openreview.net/forum?id=qMm7tC1zvj) (NeurIPS 2025 poster)
- **Cited:** NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration
- **DOI:** [10.1109/icra57147.2024.10610665](https://doi.org/10.1109/icra57147.2024.10610665)
- **Year:** 2024
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> The controller [56,58], usually a small neural network, finds a trajectory from the current place to the sub-goal and controls the robot's motion in a real-time manner.

### Surrounding context
**before:**  
> The LLM planner generates sub-goals to reach the final goal conditioned on the environment and instructions.
**after:**  
> We focus on optimizing the LLM planner, since it is the primary efficiency bottleneck due to its long inference latency and high memory cost [76,10].

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> NoMaD does describe a learned policy/controller that is used with a topological memory to plan subgoals and guide navigation, so it supports the functional part of the claim. However, NoMaD emphasizes using high-capacity, Transformer-based goal-conditioned policies rather than a "small neural network," so citing it for the claim that the controller is "usually a small neural network" is misleading/contradicted by the paper.

**Evidence quote (passage the model used):**
> To train goal-conditioned policies for visual inputs, we follow a large body of prior work on training high-capacity policies based on the Transformer architecture.

**Top retrieved passage (raw):**
> M is represented by a graph structure with nodes corresponding to the robot's visual observations in the environment, and edges corresponding to navigable paths between two nodes, as determined by the policy's goal-conditioned distance predictions. When navigating large environments, the robot's visual observations o t may not be sufficient to plan long-horizon trajectories to the goal. Instead, the robot can use the topological map M to plan a sequence of subgoals that guide the robot to the goal.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 24/28 — `t77EZLjvd5` / `7` — **CONTRADICTS**

- **Citing paper** [`t77EZLjvd5`](https://openreview.net/forum?id=t77EZLjvd5) (NeurIPS 2025 spotlight)
- **Cited:** Towards Evaluating the Robustness of Neural Networks
- **DOI:** [10.1109/sp.2017.49](https://doi.org/10.1109/sp.2017.49)
- **Year:** 2017
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> The attack budgets, including PDG attack and CW acctack [36,7], are set of ϵ a = 1/255 in default.

### Surrounding context
**before:**  
> Implementation details.
**after:**  
> The number of steps for attacks is set as 10.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The Carlini & Wagner ("Towards Evaluating the Robustness of Neural Networks") attacks are formulated as optimization problems that use a confidence parameter κ and do not rely on a default L_infty epsilon budget of 1/255. The paper discusses setting κ (they set κ=0) rather than using an epsilon bound as the default, so using this paper as support for an ϵ=1/255 default is incorrect.

**Evidence quote (passage the model used):**
> "The parameter κ encourages the solver to find an adversarial instance x that will be classified as class t with high confidence. We set κ = 0 for our attacks..."

**Top retrieved passage (raw):**
> The parameter κ encourages the solver to find an adversarial instance x that will be classified as class t with high confidence. We set κ = 0 for our attacks but we note here that a side benefit 10 One exception: The JSMA attack [38] handles this by only setting the output value to either 0 or 255.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 25/28 — `uHGG3s0XyZ` / `16` — **CONTRADICTS**

- **Citing paper** [`uHGG3s0XyZ`](https://openreview.net/forum?id=uHGG3s0XyZ) (NeurIPS 2025 poster)
- **Cited:** Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo Matching
- **DOI:** [10.1109/iccv51070.2023.00307](https://doi.org/10.1109/iccv51070.2023.00307)
- **Year:** 2023
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Surrounding context
**before:**  
> Stereo matching is a pixel-wise labeling task that relies on dense correspondence between a pair of images. The SOTA methods are either GRU-based iterative methods or Transformer-based methods.
**after:**  
> They have achieved great performance in both benchmark and zero-shot generalization testing.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper proposes an uncertainty-guided adaptive warping approach (a warping-based stereo method) rather than a GRU-based iterative disparity-update method. The retrieved passages describe the authors' warping approach and list other iterative/RAFT-style methods separately, but do not present this paper as a GRU-based iterative update method. Therefore it is not an appropriate source to back the specific claim that it uses a GRU framework to iteratively predict disparity updates.

**Evidence quote (passage the model used):**
> "iRaftStereo RVC [24] ... raft+ RVC [39] ... CREStereo++ RVC (ours) ... (a) left image (b) disparity image (c) traditional warping (d) our warping"

**Top retrieved passage (raw):**
> Stereo matching is a fundamental computer vision task [31] that aims to estimate the disparity between two rec-arXiv:2307.14071v1 [cs.CV] 26 Jul 2023 tified stereo images. In the past decade, stereo matching has become increasingly popular due to the development of deep learning and the support of large synthetic datasets [26,4]. As a result, it has a breadth of applications spanning autonomous driving [6] to 3D reconstruction [13].

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 26/28 — `uHGG3s0XyZ` / `5` — **CONTRADICTS**

- **Citing paper** [`uHGG3s0XyZ`](https://openreview.net/forum?id=uHGG3s0XyZ) (NeurIPS 2025 poster)
- **Cited:** MoCha-Stereo: Motif Channel Attention Network for Stereo Matching
- **DOI:** [10.1109/cvpr52733.2024.02623](https://doi.org/10.1109/cvpr52733.2024.02623)
- **Year:** 2024
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Surrounding context
**before:**  
> Stereo matching is a pixel-wise labeling task that relies on dense correspondence between a pair of images. The SOTA methods are either GRU-based iterative methods or Transformer-based methods.
**after:**  
> They have achieved great performance in both benchmark and zero-shot generalization testing.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> MoCha-Stereo describes a motif channel attention stereo matching network (an attention-based architecture) rather than a GRU-based iterative disparity-update method. Citing it as a GRU-iteration method is therefore incorrect.

**Evidence quote (passage the model used):**
> To address the above problems, we propose Motif Channel Attention Stereo Matching Network (MoCha-Stereo). The core idea of MoCha-Stereo is to restore the lost detailed features in feature channels by utilizing the repeated geometric contours within normal channels.

**Top retrieved passage (raw):**
> Traditional stereo matching relies on global [9], semiglobal [14], or local [2] grayscale relationships between left and right view pixels. These methods struggle to fully leverage scene-specific prior knowledge. Achieving optimal results often involves human observation, this tuning process can be resource-intensive in scenes with complex images [42]. With the advancement of deep learning, learningbased methods [16,31,41] have generally achieved better results.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 27/28 — `uHGG3s0XyZ` / `10` — **CONTRADICTS**

- **Citing paper** [`uHGG3s0XyZ`](https://openreview.net/forum?id=uHGG3s0XyZ) (NeurIPS 2025 poster)
- **Cited:** Neural Markov Random Field for Stereo Matching
- **DOI:** [10.1109/cvpr52733.2024.00522](https://doi.org/10.1109/cvpr52733.2024.00522)
- **Year:** 2024
- **Resolved by:** crossref · **Full text from:** s2_fallback

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Surrounding context
**before:**  
> Stereo matching is a pixel-wise labeling task that relies on dense correspondence between a pair of images. The SOTA methods are either GRU-based iterative methods or Transformer-based methods.
**after:**  
> They have achieved great performance in both benchmark and zero-shot generalization testing.

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The cited paper (10) presents a neural MRF approach for stereo matching, not a GRU-based iterative disparity-update method. The paper's own summary describes a data-driven MRF model and places the work in cost-volume / MRF-style approaches (not GRU iterative refinement), so it does not support being cited as an example of GRU-based iterative disparity prediction.

**Evidence quote (passage the model used):**
> "We introduce a novel fully data-driven MRF model for stereo matching..." (Introduction);
> "...cost volume aggregation [6,10,11,17,22,42,43,54,56,61,62], and iterative disparity refinement [5, 21-23, 27, 29, 30, 33, 48, 52, 55, 56]." (Related Work)

**Top retrieved passage (raw):**
> Stereo matching is a critical component in computer vision, mimicking human binocular vision to cognize 3D information within the field of view [37]. Given a pair of images, it aims to determine the horizontal displacement of individual pixels in one image to align with the other. Stereo matching has been applied to many fields like 3D reconstruction [49], autonomous navigation [53], and augmented reality [59], bridging the gap between digital imagery and real-worlds.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_

---

## Case 28/28 — `z1Cvcovlms` / `12` — **CONTRADICTS**

- **Citing paper** [`z1Cvcovlms`](https://openreview.net/forum?id=z1Cvcovlms) (NeurIPS 2025 poster)
- **Cited:** The Pushshift Reddit Dataset
- **DOI:** [10.1609/icwsm.v14i1.7347](https://doi.org/10.1609/icwsm.v14i1.7347)
- **Year:** 2020
- **Resolved by:** crossref · **Full text from:** unpaywall

### Citing sentence
> Among them, REDDIT [12] contains the comments of 50 high-quality subreddits from the REDDIT PushShift data dumps (from 2006 to 2023), and we use the data collected after 2020 to ensure the up-to-dateness.

### Surrounding context
**before:**  
> Specifically, we have observed a considerable similarity among LLM tasks in edge computing systems through extensive experiments. Figure 1 shows the similarity proportions within and across seven widely used datasets for LLM inference in edge computing systems.
**after:**  
> MMChat

### System verdict & evidence
**Verdict:** `CONTRADICTS`

**Explanation (LLM reasoning):**
> The Pushshift Reddit paper documents a complete dump of Reddit submissions and comments spanning June 2005 to April 2019 and reports billions of comments across millions of subreddits. It does not describe a curated set of "50 high-quality subreddits," nor does it cover data through 2023; therefore it does not support the citing sentence's claim about 50 subreddits or the 2006–2023 timeframe (or using data after 2020).

**Evidence quote (passage the model used):**
> Pushshift makes available all the submissions and comments posted on Reddit between June 2005 and April 2019. The dataset consists of 651,778,198 submissions and 5,601,331,385 comments posted on 2,888,885 subreddits.

**Top retrieved passage (raw):**
> Having already been used in over 100 papers from numerous disciplines over the past four years, the Pushshift Reddit dataset will continue to be a valuable resource for the research community in the future.

### Human review
- [ ] Verdict correct (MISREP confirmed)
- [ ] False alarm (citation is faithful)
- [ ] Unclear (need to read full source)

**Notes:** _(your reasoning here)_
