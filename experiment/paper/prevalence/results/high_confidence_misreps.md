# High-confidence citation misrepresentations


## Case 4/14 — `T4qJuQCFAK` / ref `[10]`

**Citing paper (NeurIPS 2025):** [T4qJuQCFAK](https://openreview.net/forum?id=T4qJuQCFAK)
**Cited paper:** Orthogonal Gradient Descent for Continual Learning
  **Year:** 2019

**Bibliography entry [N] from the citing PDF (sanity-check that we resolved the right paper):**
> Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. In International Conference on Artificial Intelligence and Statistics, pages 3762–3773. PMLR, 2020.
  _alignment overlap with our resolved title: 1.0_

### Citing sentence
> Notably, the saliency of weights is determined by analyzing the activation space (sensitivities) of the base model rather than just regularizing based on the weight space, similar to [10,26].

_(This reference is cited in 2 sentences; the verdict above is on this sentence; we list all below.)_
  - alt sentence 2: An alternative way of choosing the importance scores for changing the model weights-which has been used, e.g., in continual learning [10,26], out-of-distribution detection…

### Surrounding context (from the citing PDF)
> large performance degradation across numerous tasks while still allowing for capturing each FT expert’s capabilities. AIM seeks to achieve this by relaxing the changes to the salient weights of the base model in the final merged model. In this way, AIM is analogous to weight regularization in many continual learning approaches [1, 22, 31, 33, 39]. Notably, the saliency of weights is determined by analyzing the activation space (sensitivities) of the base model rather than just regularizing based on the weight space, similar to [10, 26]. 3.2 Activation Space Analysis AIM determines the saliency of a model’s weights by looking at the scale of activations by passing a calibration dataset to the model and recording the scale of activations in each channel. To better understand why, we will analyze how perturbation of the weights of a given model affects the model outputs. Let the or

### Primary system verdict (gpt-4o-mini, retrieved-passage RAG)
**`CONTRADICTS`**

OGD is described as a parameter-space/gradient-projection method that restricts gradient directions to avoid interference with previous tasks; it does not present an approach that determines weight saliency via activation-space sensitivity analysis. Citing OGD as similar to an activation-space saliency method is therefore misleading.

**Passage retrieved from the cited paper used as evidence:**
> we propose to address this issue from a parameter space perspective and study an approach to restrict the direction of the gradient updates to avoid forgetting previously-learned data. We present the Orthogonal Gradient Descent (OGD) method, which accomplishes this goal by projecting the gradients from new tasks onto a subspace in which the neural network output on previous task does not change...

### Second-opinion verdict (gpt-4o, full cited paper + citing context)
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not discuss determining the saliency of weights by analyzing the activation space. Instead, it focuses on Orthogonal Gradient Descent (OGD), which involves projecting gradients to avoid interference with previous tasks. The paper emphasizes preserving knowledge by maintaining orthogonal gradient directions, not by analyzing activation spaces or sensitivities as claimed in the citing sentence.


###What the AIM paper actually claims
The citing sentence says:

"the saliency of weights is determined by analyzing the activation space (sensitivities) of the base model rather than just regularizing based on the weight space, similar to [10, 26]"

The claim is that [10] (OGD) and [26] (OGD-RLS) are similar to AIM in using activation space / sensitivities for determining weight saliency — i.e., going beyond pure weight-space regularization.
What OGD actually does
OGD addresses catastrophic forgetting purely from a parameter space perspective by projecting gradients from new tasks onto a subspace in which the neural network output on previous tasks does not change. It operates entirely in gradient/weight space — there is no activation-space analysis involved anywhere. SciSpacearxiv
Now, there's one subtlety: OGD works by preserving neural network outputs (function values) rather than weights directly, which is a form of output-space sensitivity. You could charitably read "activation space / sensitivities" as loosely gesturing at this — the orthogonal projection preserves what the network computes, not just what its weights look like. But that's a very loose reading, and it's not how AIM uses activations at all (AIM literally passes data through the model and records channel magnitudes).
The real story: [26] is the relevant one, [10] is the loose one
Reference [26] — "One-pass learning via bridging OGD and recursive least-squares" — is by Navid Azizan, who is also a co-author of the AIM paper itself. The citation is clearly motivated by that connection. OGD-RLS is closer to incorporating output sensitivity. [10] (plain OGD) is the weaker half of the pair.
---



## Case 12/14 — `uHGG3s0XyZ` / ref `[10]`

**Citing paper (NeurIPS 2025):** [uHGG3s0XyZ](https://openreview.net/forum?id=uHGG3s0XyZ)
**Cited paper:** Neural Markov Random Field for Stereo Matching
  **DOI:** [10.1109/cvpr52733.2024.00522](https://doi.org/10.1109/cvpr52733.2024.00522)
  **Year:** 2024

**Bibliography entry [N] from the citing PDF (sanity-check that we resolved the right paper):**
> Guan Tongfan, Wang Chen, Liu Yun-Hui. Neural Markov Random Field for Stereo Matching // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2024. 5459–5469.
  _alignment overlap with our resolved title: 1.0_

### Citing sentence
> The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27,24,16,48,5,10].

### Surrounding context (from the citing PDF)
> and the Booster dataset demonstrate that our method achieves SOTA performance under a wide range of 3D visual illusions. 2 Related Work 2.1 Stere Matching Stereo matching is a pixel-wise labeling task that relies on dense correspondence between a pair of images. The SOTA methods are either GRU-based iterative methods or Transformer-based methods. The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27, 24, 16, 48, 5, 10]. They have achieved great performance in both benchmark and zero-shot generalization testing. The latter methods use a Transformer to learn matching and predict the disparity map [25, 12, 43, 44, 41, 23, 39]. The Transformer-based methods achieve superior performance by learning from large-scale data. In this paper, we collect a comprehensive,

### Primary system verdict (gpt-4o-mini, retrieved-passage RAG)
**`CONTRADICTS`**

The cited paper (10) presents a neural MRF approach for stereo matching, not a GRU-based iterative disparity-update method. The paper's own summary describes a data-driven MRF model and places the work in cost-volume / MRF-style approaches (not GRU iterative refinement), so it does not support being cited as an example of GRU-based iterative disparity prediction.

**Passage retrieved from the cited paper used as evidence:**
> "We introduce a novel fully data-driven MRF model for stereo matching..." (Introduction);
> "...cost volume aggregation [6,10,11,17,22,42,43,54,56,61,62], and iterative disparity refinement [5, 21-23, 27, 29, 30, 33, 48, 52, 55, 56]." (Related Work)

### Second-opinion verdict (gpt-4o, full cited paper + citing context)
**`NOT_SUPPORTED`** (high confidence)

The cited paper does not mention the use of a GRU framework for predicting disparity updates or iteratively approximating the GT value. Instead, it focuses on a Neural MRF model and a Disparity Proposal Network for stereo matching. The paper discusses neural message passing and embedding mean-field inference but does not reference GRU-based methods or iterative approximation of GT values.



✅ This is a REAL citation error — and the system is directionally correct
What the cited paper actually is
NMRF proposes a fully data-driven neural MRF model for stereo matching, where both potential functions and message passing are designed using neural networks, built on variational inference theory with a Disparity Proposal Network for tractable inference. It is an MRF/global optimisation approach, explicitly positioned as an alternative to GRU-based iterative methods. arxivarXiv
What the citing sentence claims

"The former methods predict the disparity update and iteratively approximate the GT value in a GRU framework [27, 24, 16, 48, 5, 10]."

NMRF is not a GRU-based iterative method. It doesn't predict disparity updates in a recurrent loop — it performs neural message passing in an MRF graph. NMRF's own related work section explicitly categorises "iterative disparity refinement" as a separate class of methods from its own approach, making it clear the authors of NMRF themselves draw this distinction. arXiv
Why this is genuinely wrong, unlike the previous multi-citation false positives
Unlike Cases 2, 5, 9 where the cited paper was loosely thematically relevant to the cluster, here the citation is categorically wrong — NMRF is not just a loose fit, it is the opposite paradigm from what's being described. The citing paper is building a taxonomy (GRU-based vs. Transformer-based methods), and places NMRF in the GRU bucket when NMRF itself rejects that categorisation. This is a meaningful misattribution that could mislead readers about the stereo matching landscape.
This is likely a copy-paste error in the citing paper — someone assembled the citation list for GRU-based methods and accidentally included NMRF, perhaps confusing it with another iterative stereo method.
One nuance the system got slightly wrong
The system's framing is correct, but its evidence ("MRF model... not GRU") is from the abstract. The stronger evidence is NMRF's own related work section explicitly separating itself from GRU iterative refinement methods. The system happened to land on the right verdict for the right general reason, even if the retrieved passage wasn't the most damning one.
---



