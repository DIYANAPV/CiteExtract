# Second-opinion review — 49 misrep cases

Primary system flagged these as MISREP. A second LLM (gpt-4o) with full citing context + full cited paper agreed with the primary system on **24/49** cases (49.0%). 25 cases the second LLM judged SUPPORTED (potential false positives). 0 could not be evaluated (no full text available).

---

## Case 1/28 — `0Nq1GvAGR0` / `30`

- Cited: Attention Based Vehicle Trajectory Prediction
- DOI: [10.1109/TIV.2020.2991952](https://doi.org/10.1109/TIV.2020.2991952)
- Cited text source: `oa_url` (43972 chars)

### Citing sentence
> (ii) knowledge-free methods, which rely on deep learning's ability to encode complex data features, modeling them using structures such as tensors [29,30] or attention mechanisms

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper presents an attention-based trajectory prediction method and repeatedly emphasizes attention mechanisms and modeling vehicle interactions. It does not describe tensor-based modeling as the sentence attributes to it, so it is not an appropriate citation for 'modeling ... using structures such as tensors.'

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The cited paper discusses the use of attention mechanisms for vehicle trajectory prediction, which aligns with the citing sentence's mention of 'attention mechanisms' as part of knowledge-free methods. The cited paper explicitly states that it uses 'an attention mechanism that explicitly highlights the importance of neighboring vehicles with respect to their future states,' supporting the claim that it relies on deep learning to model complex data features without prior knowledge.

**Agree with primary?** ❌ no

---

## Case 2/28 — `59TL8MpBzX` / `7`

- Cited: MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction
- DOI: [10.1109/cvprw56347.2022.00090](https://doi.org/10.1109/cvprw56347.2022.00090)
- Cited text source: `s2_fallback` (25349 chars)

### Citing sentence
> Comparing Variant 1 with SimDiff-Spectra (i.e., a UNet-based HSI reconstruction network, which is similar to [7]), we find that the AICD primarily contributes to the spatial details recovery of the HSI signal, which aligns with our original design intention.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — MST++ is presented as a Transformer-based framework (not a UNet/CNN-based network). Citing MST++ as an example of a UNet-based method is inconsistent with MST++'s description, so this citation is inappropriate.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that SimDiff-Spectra is a 'UNet-based HSI reconstruction network, which is similar to [7]'. However, the cited paper [7] describes MST++, a Transformer-based framework for spectral reconstruction, not a UNet-based method. The cited paper does not mention UNet or describe a network similar to SimDiff-Spectra as claimed.

**Agree with primary?** ✅ yes

---

## Case 3/28 — `5xwyxupsLL` / `31`

- Cited: One-Step Diffusion with Distribution Matching Distillation
- DOI: [10.1109/cvpr52733.2024.00632](https://doi.org/10.1109/cvpr52733.2024.00632)
- Cited text source: `s2_fallback` (36432 chars)

### Citing sentence
> For models with only a handful of steps-such as distilled variants like the 4-step Flux.1-schnell [31] or one-step diffusion methods [32,33]-temporal redundancy is minimal, making PipeFusion less effective.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (One-Step Diffusion with Distribution Matching Distillation) presents a distillation pipeline that reduces the multi-step generative process to a single forward pass (i.e., one-step diffusion), not a 4-step distilled variant. Thus it does not support the specific claim that it is an example of a "4-step Flux.1-schnell" model; the citation is misattributed. While the paper is relevant as an example of small-step/one-step distilled models (which is topically related), it contradicts the specific labeling as a 4-step variant.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The cited paper discusses methods to accelerate diffusion models by reducing the number of steps required for image generation, including one-step diffusion methods. It mentions that distilled models, which are simplified versions of multi-step diffusion models, have minimal temporal redundancy. This aligns with the citing sentence's claim that for models with only a few steps, like one-step diffusion methods, temporal redundancy is minimal, making PipeFusion less effective.

**Agree with primary?** ❌ no

---

## Case 4/28 — `AmZ7uHDJiR` / `7`

- Cited: Neural Topological SLAM for Visual Navigation
- DOI: [10.1109/cvpr42600.2020.01289](https://doi.org/10.1109/cvpr42600.2020.01289)
- Cited text source: `unpaywall` (34286 chars)

### Citing sentence
> Most SLAM systems [38,34,7,58,60,56,49,20,27,16] assume an autonomous agent navigating an environment with distant, static illumination, e.g., a self-driving car in the streets, and they optimize a Photometric Bundle Adjustment loss where they minimize an error between the captured image and the re-rendered image using estimated 3D scene and camera poses.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (Neural Topological SLAM, NTS) describes a topological SLAM approach that builds a graph of panoramic images and relative poses for navigation, not a metric photometric bundle-adjustment system that minimizes image vs. re-rendered image photometric error. Thus it does not support the claim that 'most SLAM systems' optimize a photometric BA loss; in fact NTS exemplifies a different (topological) SLAM paradigm and is not evidence for the photometric BA assertion.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not discuss SLAM systems assuming distant, static illumination or optimizing a Photometric Bundle Adjustment loss. Instead, it focuses on a topological representation for navigation using a graph-based approach with nodes represented by panoramic images and functions for geometric and semantic predictions. There is no mention of minimizing errors between captured and re-rendered images using estimated 3D scenes and camera poses.

**Agree with primary?** ✅ yes

---

## Case 5/28 — `AuOZDp4gy7` / `36`

- Cited: TS2Vec: Towards Universal Representation of Time Series
- DOI: [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- Cited text source: `unpaywall` (25658 chars)

### Citing sentence
> The best results are highlighted in bold.0.625±0.001.391±0.001 0.044±0.002 0.429±0.001 2.010±0.003
> Table 18 :18Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The sentence uses [36] to refer to TimesNet. The cited paper (36) is TS2Vec, not TimesNet, so the citation does not support the claim and appears to be incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The citing sentence refers to the results of the proposed AliO method on various datasets, including TimesNet [36], and reports metrics such as MSE, TAM, MAE, MAPE, and RMSE. The cited paper discusses the TS2Vec framework, which is evaluated on time series tasks including forecasting, and reports improvements in performance metrics. The cited paper supports the claim as it provides results on time series forecasting, which aligns with the metrics mentioned in the citing sentence.

**Agree with primary?** ❌ no

---

## Case 6/28 — `AuOZDp4gy7` / `36`

- Cited: TS2Vec: Towards Universal Representation of Time Series
- DOI: [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- Cited text source: `unpaywall` (25658 chars)

### Citing sentence
> The best results are highlighted in bold.0.625±0.001.391±0.001 0.044±0.002 0.429±0.001 2.010±0.003
> Table 18 :18Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The sentence uses [36] to refer to TimesNet. The cited paper (36) is TS2Vec, not TimesNet, so the citation does not support the claim and appears to be incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The citing sentence refers to the results of the proposed AliO method on various datasets, including TimesNet [36], and reports metrics such as MSE, TAM, MAE, MAPE, and RMSE. The cited paper discusses the TS2Vec framework, which is evaluated on time series tasks including forecasting, and reports improvements in performance metrics. The cited paper supports the claim as it provides results on time series forecasting, which aligns with the metrics mentioned in the citing sentence.

**Agree with primary?** ❌ no

---

## Case 7/28 — `AuOZDp4gy7` / `36`

- Cited: TS2Vec: Towards Universal Representation of Time Series
- DOI: [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- Cited text source: `unpaywall` (25658 chars)

### Citing sentence
> The best results are highlighted in bold.0.625±0.001.391±0.001 0.044±0.002 0.429±0.001 2.010±0.003
> Table 18 :18Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The sentence uses [36] to refer to TimesNet. The cited paper (36) is TS2Vec, not TimesNet, so the citation does not support the claim and appears to be incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The citing sentence refers to the results of the proposed AliO method on various datasets, including TimesNet [36], and reports metrics such as MSE, TAM, MAE, MAPE, and RMSE. The cited paper discusses the TS2Vec framework, which is evaluated on time series tasks including forecasting, and reports improvements in performance metrics. The cited paper supports the claim as it provides results on time series forecasting, which aligns with the metrics mentioned in the citing sentence.

**Agree with primary?** ❌ no

---

## Case 8/28 — `AuOZDp4gy7` / `36`

- Cited: TS2Vec: Towards Universal Representation of Time Series
- DOI: [10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)
- Cited text source: `unpaywall` (25658 chars)

### Citing sentence
> The best results are highlighted in bold.0.625±0.001.391±0.001 0.044±0.002 0.429±0.001 2.010±0.003
> Table 18 :18Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Traffic, ILI, and Weather datasets and TimesNet[36].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The sentence uses [36] to refer to TimesNet. The cited paper (36) is TS2Vec, not TimesNet, so the citation does not support the claim and appears to be incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The citing sentence refers to the results of the proposed AliO method on various datasets, including TimesNet [36], and reports metrics such as MSE, TAM, MAE, MAPE, and RMSE. The cited paper discusses the TS2Vec framework, which is evaluated on time series tasks including forecasting, and reports improvements in performance metrics. The cited paper supports the claim as it provides results on time series forecasting, which aligns with the metrics mentioned in the citing sentence.

**Agree with primary?** ❌ no

---

## Case 9/28 — `AuOZDp4gy7` / `9`

- Cited: Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- DOI: [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- Cited text source: `s2_fallback` (55144 chars)

### Citing sentence
> We
> Table 23 :23Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — Duplicate of the previous table citation: the paper being cited is about self-supervised contrastive methods for semi-supervised classification, not GPT4TS. The attribution therefore appears incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that the results of the proposed AliO method on various datasets and GPT4TS are reported in terms of MSE, TAM, MAE, MAPE, and RMSE. However, the cited paper does not mention the AliO method or provide results in terms of these metrics for GPT4TS. The cited paper focuses on a novel framework for time-series representation learning using contrastive learning, without discussing the specific metrics or methods mentioned in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 10/28 — `AuOZDp4gy7` / `9`

- Cited: Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- DOI: [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- Cited text source: `s2_fallback` (55144 chars)

### Citing sentence
> We
> Table 23 :23Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — Duplicate of the previous table citation: the paper being cited is about self-supervised contrastive methods for semi-supervised classification, not GPT4TS. The attribution therefore appears incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that the results of the proposed AliO method on various datasets and GPT4TS are reported in terms of MSE, TAM, MAE, MAPE, and RMSE. However, the cited paper does not mention the AliO method or provide results in terms of these metrics for GPT4TS. The cited paper focuses on a novel framework for time-series representation learning using contrastive learning, without discussing the specific metrics or methods mentioned in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 11/28 — `AuOZDp4gy7` / `9`

- Cited: Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- DOI: [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- Cited text source: `s2_fallback` (55144 chars)

### Citing sentence
> We
> Table 23 :23Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — Duplicate of the previous table citation: the paper being cited is about self-supervised contrastive methods for semi-supervised classification, not GPT4TS. The attribution therefore appears incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that the results of the proposed AliO method on various datasets and GPT4TS are reported in terms of MSE, TAM, MAE, MAPE, and RMSE. However, the cited paper does not mention the AliO method or provide results in terms of these metrics for GPT4TS. The cited paper focuses on a novel framework for time-series representation learning using contrastive learning, without discussing the specific metrics or methods mentioned in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 12/28 — `AuOZDp4gy7` / `9`

- Cited: Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification
- DOI: [10.1109/tpami.2023.3308189](https://doi.org/10.1109/tpami.2023.3308189)
- Cited text source: `s2_fallback` (55144 chars)

### Citing sentence
> We
> Table 23 :23Full results of the proposed AliO method on the ECL, ETTh1, ETTh2, ETTm1, ETTm2, ILI, Traffix, and Weather datasets and GPT4TS[43].The results are reported in terms of MSE, TAM, MAE, MAPE, and RMSE.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — Duplicate of the previous table citation: the paper being cited is about self-supervised contrastive methods for semi-supervised classification, not GPT4TS. The attribution therefore appears incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that the results of the proposed AliO method on various datasets and GPT4TS are reported in terms of MSE, TAM, MAE, MAPE, and RMSE. However, the cited paper does not mention the AliO method or provide results in terms of these metrics for GPT4TS. The cited paper focuses on a novel framework for time-series representation learning using contrastive learning, without discussing the specific metrics or methods mentioned in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 13/28 — `T4qJuQCFAK` / `10`

- Cited: Orthogonal Gradient Descent for Continual Learning
- Cited text source: `arxiv` (35921 chars)

### Citing sentence
> Notably, the saliency of weights is determined by analyzing the activation space (sensitivities) of the base model rather than just regularizing based on the weight space, similar to [10,26].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — OGD is described as a parameter-space/gradient-projection method that restricts gradient directions to avoid interference with previous tasks; it does not present an approach that determines weight saliency via activation-space sensitivity analysis. Citing OGD as similar to an activation-space saliency method is therefore misleading.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not discuss determining the saliency of weights by analyzing the activation space. Instead, it focuses on Orthogonal Gradient Descent (OGD), which involves projecting gradients to avoid interference with previous tasks. The paper emphasizes preserving knowledge by maintaining orthogonal gradient directions, not by analyzing activation spaces or sensitivities as claimed in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 14/28 — `T4qJuQCFAK` / `10`

- Cited: Orthogonal Gradient Descent for Continual Learning
- Cited text source: `arxiv` (35921 chars)

### Citing sentence
> Notably, the saliency of weights is determined by analyzing the activation space (sensitivities) of the base model rather than just regularizing based on the weight space, similar to [10,26].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — OGD is described as a parameter-space/gradient-projection method that restricts gradient directions to avoid interference with previous tasks; it does not present an approach that determines weight saliency via activation-space sensitivity analysis. Citing OGD as similar to an activation-space saliency method is therefore misleading.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not discuss determining the saliency of weights by analyzing the activation space. Instead, it focuses on Orthogonal Gradient Descent (OGD), which involves projecting gradients to avoid interference with previous tasks. The paper emphasizes preserving knowledge by maintaining orthogonal gradient directions, not by analyzing activation spaces or sensitivities as claimed in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 15/28 — `fru52tkjHf` / `8`

- Cited: signSGD: compressed optimisation for non-convex problems
- Cited text source: `arxiv` (48475 chars)

### Citing sentence
> [Bernstein et al., 2018], may fail to converge [Karimireddy et al., 2019].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The signSGD paper presents convergence results and convergence-rate theorems for SIGNSGD/SIGNUM under its assumptions and argues SIGNSGD achieves SGD-level convergence; it does not state that SignSGD may fail to converge in the general sense. Using Bernstein et al. to support the claim that SignSGD "may fail to converge" is therefore not supported by the cited paper (and in fact the paper offers positive convergence results), so the citation is inappropriate for that negative claim.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The cited paper by Bernstein et al. (2018) discusses the convergence properties of SIGNSGD, noting that it can fail to converge under certain conditions. Specifically, the paper states that 'the sign of the stochastic gradient is a biased approximation to the true gradient, making it more challenging to analyse compared to standard SGD.' This aligns with the citing sentence's claim that SIGNSGD may fail to converge.

**Agree with primary?** ❌ no

---

## Case 16/28 — `fru52tkjHf` / `24`

- Cited: AURA: An Adaptive Unified Regularization Approach for Gradient-Based Optimization
- DOI: [10.21203/rs.3.rs-7480833/v1](https://doi.org/10.21203/rs.3.rs-7480833/v1)
- Cited text source: `unpaywall` (9800 chars)

### Citing sentence
> [Gupta et al., 2017] and mainly highlight its superior convergence results compared to Shampoo and fullmatrix AdaGrad.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (AURA) presents a new optimizer called AURA and compares it experimentally to Adam, AdamW, AMSGrad, and RMSProp. It does not mention the AdaptReg unified preconditioning framework, nor does it report comparisons against Shampoo or full-matrix AdaGrad. Thus it does not support the claim that the authors developed their method from AdaptReg or that the paper 'mainly highlight[s]' superiority over Shampoo/full-matrix AdaGrad — the attribution is incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper by Gupta et al. (2017) does not mention or compare the convergence results of their method with Shampoo or full-matrix AdaGrad. The paper focuses on introducing the AURA optimizer, which is a novel approach that adapts momentum dynamically, and compares it with other optimizers like Adam, AdamW, RMSProp, and AMSGrad. There is no mention of Shampoo or full-matrix AdaGrad in the context of convergence results.

**Agree with primary?** ✅ yes

---

## Case 17/28 — `g42mGfR6We` / `40`

- Cited: Prompt Guided Transformer for Multi-Task Dense Prediction
- DOI: [10.1109/tmm.2024.3349865](https://doi.org/10.1109/tmm.2024.3349865)
- Cited text source: `s2_fallback` (35029 chars)

### Citing sentence
> Focusing on this problem, many adapter-based methods have been proposed [37,50,57,65,40].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (PGT) presents a prompt-guided, task-conditional transformer that uses prompting / learnable task prompts as a param-efficient adaptation method, not traditional adapter modules. The paper frames its approach as prompting / task-conditioned prompts rather than inserting adapter layers, so treating it as an "adapter-based" method is a misattribution.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that many adapter-based methods have been proposed, but the cited paper does not mention adapter-based methods. Instead, it discusses multi-task learning (MTL) methods, including encoder-focused, decoder-focused, and task-conditional architectures. There is no mention of adapter-based methods in the cited paper, which focuses on a novel method called Prompt Guided Transformer (PGT) for multi-task dense prediction.

**Agree with primary?** ✅ yes

---

## Case 18/28 — `kND7h1kD53` / `26`

- Cited: A Euclidean transformer for fast and stable machine learned force fields
- DOI: [10.1038/s41467-024-50620-6](https://doi.org/10.1038/s41467-024-50620-6)
- Cited text source: `unpaywall` (59997 chars)

### Citing sentence
> Unlike existing methods which either use nonlocal attention using spherical harmonics [26] in Euclidean-space neighborhoods or using high-degree steerable representations within a single geometric space [17,6], DualEquiNet jointly constructs complementary representations in both Euclidean (EU) and spherical harmonics (SH) spaces: the EU space captures local geometric interactions crucial for atomiclevel detail, while the SH space encodes rotation-equivariant features well-suited for long-range structural dependencies.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The citing sentence attributes to [26] the use of "nonlocal attention using spherical harmonics in Euclidean-space neighborhoods." The cited paper instead emphasizes a Euclidean self-attention mechanism that replaces SO(3) convolutions (i.e., it is proposing an alternative to spherical-harmonic / SO(3)-convolution approaches), so it does not support the described category. The paper presents Euclidean self-attention / SO3KRATES rather than a spherical-harmonics nonlocal-attention method, making the citation inappropriate for that specific claim.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence claims that the cited paper [17] uses high-degree steerable representations within a single geometric space. However, the cited paper primarily discusses the SO3KRATES method, which employs a Euclidean self-attention mechanism and avoids expensive SO(3) convolutions by using a filter on the relative orientation of atomic neighborhoods. It does not focus on high-degree steerable representations within a single geometric space as described in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 19/28 — `kND7h1kD53` / `65`

- Cited: E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction
- Cited text source: `oa_url` (43888 chars)

### Citing sentence
> 3D RNA structures are generated using RhoFold [65] following prior works [57].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (reference [65]) is E2Efold-3D, which introduces a different method named E2Efold-3D for de novo RNA 3D structure prediction. It does not present or document a tool called RhoFold, so citing it as the source for RhoFold is incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not mention RhoFold or its use in generating 3D RNA structures. Instead, it introduces E2Efold-3D, a deep learning-based method for RNA 3D structure prediction. The paper discusses the challenges of RNA structure determination and the development of computational methods, but RhoFold is not referenced or described as part of their methodology.

**Agree with primary?** ✅ yes

---

## Case 20/28 — `kND7h1kD53` / `79`

- Cited: Shape-aware Graph Spectral Learning
- DOI: [10.1145/3627673.3679604](https://doi.org/10.1145/3627673.3679604)
- Cited text source: `s2_fallback` (41872 chars)

### Citing sentence
> A variety of benchmarks [24,70,86,79] have been proposed to study this problem, along with several representative solutions, including Implicit Graph Neural Networks (IGNNs) and Graph Transformers (GTs).

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (Shape-aware Graph Spectral Learning) is a methods paper that analyzes spectral filters and proposes a regularization and a model (NewtonNet), with experiments on existing datasets. It does not present itself as a benchmark or a benchmark suite proposed to study long-range dependency problems. Thus it is not an appropriate citation for the claim that it is a proposed benchmark.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not discuss benchmarks for studying long-range dependencies in GNNs, nor does it mention Implicit Graph Neural Networks (IGNNs) or Graph Transformers (GTs). The paper focuses on spectral GNNs and their adaptation to homophily ratios, without addressing the specific problem of long-range dependencies or the mentioned solutions.

**Agree with primary?** ✅ yes

---

## Case 21/28 — `kND7h1kD53` / `65`

- Cited: E2Efold-3D: End-to-End Deep Learning Method for accurate de novo RNA 3D Structure Prediction
- Cited text source: `oa_url` (43888 chars)

### Citing sentence
> 3D RNA structures are generated using RhoFold [65] following prior works [57].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (reference [65]) is E2Efold-3D, which introduces a different method named E2Efold-3D for de novo RNA 3D structure prediction. It does not present or document a tool called RhoFold, so citing it as the source for RhoFold is incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not mention RhoFold or its use in generating 3D RNA structures. Instead, it introduces E2Efold-3D, a deep learning-based method for RNA 3D structure prediction. The paper discusses the challenges of RNA structure determination and the development of computational methods, but RhoFold is not referenced or described as part of their methodology.

**Agree with primary?** ✅ yes

---

## Case 22/28 — `kND7h1kD53` / `74`

- Cited: Deep learning models for predicting RNA degradation via dual crowdsourcing
- DOI: [10.1038/s42256-022-00571-8](https://doi.org/10.1038/s42256-022-00571-8)
- Cited text source: `html_meta_pdf` (33096 chars)

### Citing sentence
> ModelTrain Time (s/epoch) Memory (MB)SchNet4.361628EGNN6.240608GVPGNN11.17210160TFN56.23543970FastEGNN11.7551954HEGNN6.3201232GotenNet10.3325894DualEqui10.9902436D.6 Datasets DetailsCOVID Vaccine The OpenVaccine COVID-19 dataset [74] consists of 4,082 synthetic RNAsequences, each 107-130 nucleotides long.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper reports a different dataset size and length range: the OpenVaccine Kaggle dataset comprises 6,043 RNA constructs of 102–130 nucleotides (not 4,082 sequences of 107–130 nt). The available abstract and text thus contradict the specific numeric claim in the citing sentence.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The citing sentence refers to the OpenVaccine COVID-19 dataset [74] as consisting of 4,082 synthetic RNA sequences, each 107-130 nucleotides long. However, the cited paper does not mention a dataset with exactly 4,082 sequences or specify the length range of 107-130 nucleotides. The cited paper discusses the OpenVaccine Kaggle competition and mentions 3,029 RNA designs of length 107 nt collected in the first round, but does not provide details matching the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 23/28 — `qMm7tC1zvj` / `58`

- Cited: NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration
- DOI: [10.1109/icra57147.2024.10610665](https://doi.org/10.1109/icra57147.2024.10610665)
- Cited text source: `s2_fallback` (28531 chars)

### Citing sentence
> The controller [56,58], usually a small neural network, finds a trajectory from the current place to the sub-goal and controls the robot's motion in a real-time manner.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — NoMaD does describe a learned policy/controller that is used with a topological memory to plan subgoals and guide navigation, so it supports the functional part of the claim. However, NoMaD emphasizes using high-capacity, Transformer-based goal-conditioned policies rather than a "small neural network," so citing it for the claim that the controller is "usually a small neural network" is misleading/contradicted by the paper.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not describe a controller that is 'usually a small neural network' for finding a trajectory and controlling robot motion in real-time. Instead, it presents NoMaD, a diffusion model-based approach for goal-conditioned navigation and exploration, which uses a Transformer-based policy and diffusion models, not a small neural network. The paper focuses on a unified policy for both goal-directed and task-agnostic navigation, rather than a specific controller as described in the citing sentence.

**Agree with primary?** ✅ yes

---

## Case 24/28 — `t77EZLjvd5` / `7`

- Cited: Towards Evaluating the Robustness of Neural Networks
- DOI: [10.1109/sp.2017.49](https://doi.org/10.1109/sp.2017.49)
- Cited text source: `unpaywall` (63890 chars)

### Citing sentence
> The attack budgets, including PDG attack and CW acctack [36,7], are set of ϵ a = 1/255 in default.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The Carlini & Wagner ("Towards Evaluating the Robustness of Neural Networks") attacks are formulated as optimization problems that use a confidence parameter κ and do not rely on a default L_infty epsilon budget of 1/255. The paper discusses setting κ (they set κ=0) rather than using an epsilon bound as the default, so using this paper as support for an ϵ=1/255 default is incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not mention specific attack budgets or the use of ϵ = 1/255 for PDG or CW attacks. The paper discusses various adversarial attacks and their effectiveness but does not specify the attack budgets or parameters like ϵ = 1/255. The citing sentence's claim about the attack budgets is not supported by the content of the cited paper.

**Agree with primary?** ✅ yes

---

## Case 25/28 — `uHGG3s0XyZ` / `16`

- Cited: Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo Matching
- DOI: [10.1109/iccv51070.2023.00307](https://doi.org/10.1109/iccv51070.2023.00307)
- Cited text source: `s2_fallback` (25930 chars)

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper proposes an uncertainty-guided adaptive warping approach (a warping-based stereo method) rather than a GRU-based iterative disparity-update method. The retrieved passages describe the authors' warping approach and list other iterative/RAFT-style methods separately, but do not present this paper as a GRU-based iterative update method. Therefore it is not an appropriate source to back the specific claim that it uses a GRU framework to iteratively predict disparity updates.

### Second-opinion (gpt-4o, full cited paper) verdict
**`SUPPORTED`** (high confidence)

The cited paper discusses the use of a GRU framework for iterative refinement in stereo matching, specifically mentioning the use of Gate Recurrent Unit (GRU) in their Adaptive Recurrent Module (ARM) for refining disparities. This aligns with the citing sentence's claim that the methods predict disparity updates and iteratively approximate the ground truth value in a GRU framework. The cited paper states, 'The disparities are then refined with correlation using Gate Recurrent Unit (GRU).' This supports the claim made in the citing sentence.

**Agree with primary?** ❌ no

---

## Case 26/28 — `uHGG3s0XyZ` / `5`

- Cited: MoCha-Stereo: Motif Channel Attention Network for Stereo Matching
- DOI: [10.1109/cvpr52733.2024.02623](https://doi.org/10.1109/cvpr52733.2024.02623)
- Cited text source: `s2_fallback` (23731 chars)

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — MoCha-Stereo describes a motif channel attention stereo matching network (an attention-based architecture) rather than a GRU-based iterative disparity-update method. Citing it as a GRU-iteration method is therefore incorrect.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not mention the use of a GRU framework for predicting disparity updates or iteratively approximating the GT value. Instead, it focuses on the MoCha-Stereo framework, which uses Motif Channel Attention and other techniques for stereo matching. There is no reference to GRU-based methods in the cited paper.

**Agree with primary?** ✅ yes

---

## Case 27/28 — `uHGG3s0XyZ` / `10`

- Cited: Neural Markov Random Field for Stereo Matching
- DOI: [10.1109/cvpr52733.2024.00522](https://doi.org/10.1109/cvpr52733.2024.00522)
- Cited text source: `s2_fallback` (36418 chars)

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The cited paper (10) presents a neural MRF approach for stereo matching, not a GRU-based iterative disparity-update method. The paper's own summary describes a data-driven MRF model and places the work in cost-volume / MRF-style approaches (not GRU iterative refinement), so it does not support being cited as an example of GRU-based iterative disparity prediction.

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not mention the use of a GRU framework for predicting disparity updates or iteratively approximating the GT value. Instead, it focuses on a Neural MRF model and a Disparity Proposal Network for stereo matching. The paper discusses neural message passing and embedding mean-field inference but does not reference GRU-based methods or iterative approximation of GT values.

**Agree with primary?** ✅ yes

---

## Case 28/28 — `z1Cvcovlms` / `12`

- Cited: The Pushshift Reddit Dataset
- DOI: [10.1609/icwsm.v14i1.7347](https://doi.org/10.1609/icwsm.v14i1.7347)
- Cited text source: `unpaywall` (30633 chars)

### Citing sentence
> Among them, REDDIT [12] contains the comments of 50 high-quality subreddits from the REDDIT PushShift data dumps (from 2006 to 2023), and we use the data collected after 2020 to ensure the up-to-dateness.

### Primary system (gpt-4o-mini, retrieval-only) verdict
**`CONTRADICTS`** — The Pushshift Reddit paper documents a complete dump of Reddit submissions and comments spanning June 2005 to April 2019 and reports billions of comments across millions of subreddits. It does not describe a curated set of "50 high-quality subreddits," nor does it cover data through 2023; therefore it does not support the citing sentence's claim about 50 subreddits or the 2006–2023 timeframe (or using data after 2020).

### Second-opinion (gpt-4o, full cited paper) verdict
**`NOT_SUPPORTED`** (high confidence)

The cited paper describes the Pushshift Reddit dataset, which includes data from 2005 to 2019, but does not mention any data collected after 2020. The citing sentence claims that data collected after 2020 is used, which is not supported by the information provided in the cited paper.

**Agree with primary?** ✅ yes
