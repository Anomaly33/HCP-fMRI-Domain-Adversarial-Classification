# HCP-fMRI-Domain-Adversarial-Classification
PyTorch implementation of a **Domain Adversarial Neural Network (DANN)** for **face vs. shape classification** on **Human Connectome Project (HCP) fMRI data**.  
The model learns **domain-invariant** features across subjects and reaches **96.62% test accuracy** with strong ROC-AUC and PR-AUC.

---

## TL;DR
- **Task:** Face vs. Shape classification on HCP task fMRI  
- **Method:** DANN (feature extractor + label head + domain discriminator via **Gradient Reversal Layer**)  
- **Why:** Address subject distribution shift / domain shift in fMRI  
- **Results:** Val **96.88%**, Test **96.62%**, ROC-AUC **0.9922**, AP **0.9913**  
- **Goodies:** Training & evaluation scripts, pretrained checkpoints, plots, and dataset link

---

