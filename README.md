# CAT-XPLAIN: Causality for Inherently Explainable Transformers

**CAT-XPLAIN** introduces causal explanation capability directly into Vision Transformers (ViT), enabling models to *inherently* identify the most causally significant image regions — without any post-hoc explainer.

📄 **Paper:** [Causality for Inherently Explainable Transformers: CAT-XPLAIN](https://arxiv.org/abs/2206.14841)  
🎤 **Spotlight presentation** at the [Explainable AI for Computer Vision (XAI4CV) Workshop, CVPR 2022](https://xai4cv.github.io/workshop-schedule)  
🚀 **Quick demo:** [Google Colab](https://colab.research.google.com/drive/1tpzcLL1vX_mu0Pmc2Snz1ChqwX2acFXC?usp=sharing)

---

## Citation

If you use this code or our method in your work, please cite:

```bibtex
@inproceedings{khanal2022causality,
  title={Causality for Inherently Explainable Transformers: {CAT-XPLAIN}},
  author={Khanal, Subash and Brodie, Benjamin and Xing, Xin and Lin, Ai-Ling and Jacobs, Nathan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2022}
}
```

---

## Overview

This project is motivated by ["Instance-wise Causal Feature Selection for Model Interpretation" (Panda et al., CVPRW 2021)](https://openaccess.thecvf.com/content/CVPR2021W/CiV/papers/Panda_Instance-Wise_Causal_Feature_Selection_for_Model_Interpretation_CVPRW_2021_paper.pdf), which proposes a model-agnostic post-hoc explainer that identifies the most significant causal regions per input instance.

Unlike that post-hoc approach, **CAT-XPLAIN** makes a small modification to the existing Transformer architecture so that the model *inherently* identifies the most causally important regions while performing its primary classification task. This results in an interpretable Transformer that requires no separate explainer model.

---

## Requirements

- Python 3.8
- CUDA 10.2 (for GPU support; CPU-only training also works)
- Conda (recommended for environment management)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mvrl/CAT-XPLAIN.git
cd CAT-XPLAIN

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate CAT-XPLAIN
```

---

## Running Experiments

All scripts below should be run from the **repository root** (`CAT-XPLAIN/`). Checkpoints and results are saved automatically under `MNIST_FMNIST_CIFAR/checkpoints/` and `MNIST_FMNIST_CIFAR/csv_results/`.

### Post-hoc baseline (MNIST, Fashion-MNIST, CIFAR-10)

```bash
sh ./MNIST_FMNIST_CIFAR/posthoc_run.sh
```

### CAT-XPLAIN interpretable transformer (MNIST, Fashion-MNIST, CIFAR-10)

```bash
sh ./MNIST_FMNIST_CIFAR/expViT_run.sh
```

---

## Acknowledgements

This code is adapted from Pranoy Panda's repository: [Instance-wise Causal Feature Selection for Model Interpretation](https://github.com/pranoy-panda/Causal-Feature-Subset-Selection).

---

## Contact

Subash Khanal  
Washington University in St. Louis  
k.subash@wustl.edu
