# ATtention for CAusal eXPLAINation (CAT-XPLAIN)
This project aims at providing causal explaination capability for existing Neural Network Architectures. It utilizes attention mechanism of Transformers to attend and hence identify the most important regions of input that have highest Causal significance to the output. This project is motivated from a paper titled ["Instance-wise Causal Feature Selection for Model Interpretation", 2021 by Pranoy et. al.](https://openaccess.thecvf.com/content/CVPR2021W/CiV/papers/Panda_Instance-Wise_Causal_Feature_Selection_for_Model_Interpretation_CVPRW_2021_paper.pdf) 

Their paper proposes to build a model agnostic post-hoc explainer model that is able to identify the most causally significant regions in the input space (features selection) of each instance. Unlike the post-hoc explanation approach, we propose to minor modification on the existing Transformer architecture so that the model is able to inherently identify the regions with highest causal strength while performing the task they are designed for. This leads to development of Transformers with causal explaination capability without the need of additional post-hoc explainer.


### Steps

1. `git clone git@github.com:mvrl/CAT-XPLAIN.git`
2. `cd CAT-XPLAIN`
3. Create a virtual environment for the project.
    `conda env create -f environment.yml`
4.  `conda activate CAT-XPLAIN`
5. Run the post-hoc experiments for MNIST,FMNIST, and CIFAR datasets.\
    `sh ./MNIST_FMNIST_CIFAR/posthoc_run.sh`
6. Run Interpretable transformer  for MNIST,FMNIST, and CIFAR datasets.\
    `sh ./MNIST_FMNIST_CIFAR/expViT_run.sh`
