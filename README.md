# ATtention for CAusal eXPLAINation (CAT-XPLAIN)
This project aims at providing causal explaination capability for existing Neural Network Architectures. It utilizes attention mechanism of Transformers to attend and hence identify the most important regions of input that have highest Causal significance to the output. This project is motivated from a paper titled ["Instance-wise Causal Feature Selection for Model Interpretation", 2021 by Pranoy et. al.](https://openaccess.thecvf.com/content/CVPR2021W/CiV/papers/Panda_Instance-Wise_Causal_Feature_Selection_for_Model_Interpretation_CVPRW_2021_paper.pdf) 

Their paper proposes to build a model agnostic post-hoc explainer model that is able to identify the most causally significant regions in the input space (features selection) of each instance. While their approach assumes that there exist a causal relationship between the input and output space, they ignore the relationships within the input space. Our project tries to offer solution for this limitation by using Transformers which in the core are based on self-attention mechanism effectively leveraging the relationships within the input space of each instance. Moreover, Unlike the post-hoc explanation approach, we propose to modify the existing Transformer architecture so that they are able to inherently identify the regions with highest causal strength while performing the task they are designed for. This leads to development of Transformers with causal explaination capability.


### Steps (for MNIST and FMNIST)

1. `https://github.com/Subash33/CAT-XPLAIN`
2. `cd CAT-XPLAIN`
3. Create a virtual environment for the project.\
`conda env create --file environment.yml`
4. `conda activate CAT-XPLAIN`
5. Run the post-hoc experiment for MNIST or FMNIST datasets

`python post_hoc.py --num_patches 6 --validation "with_test" --bb_model_type "ViT" --sel_model_type "ViT" --dataset_name "mnist"`

`python post_hoc.py --num_patches 6 --validation "with_test" --bb_model_type "ViT" --sel_model_type "ViT" --dataset_name "fmnist"`

6. Run the Interpretable transformer for MNIST or FMNIST datasets

`python interpretable_transformer.py --num_patches 6 --validation "with_test" --loss_weight 0.9 --dataset_name "mnist"`

`python interpretable_transformer.py --num_patches 6 --validation "with_test" --loss_weight 0.9 --dataset_name "fmnist"`

### Steps (for ADNI MRI dataset)

1. Download ADNI data using the fileID provided after access request at subash.khanal33@gmail.com

`gdown "https://drive.google.com/uc?id=FILEID"`\
Note: Replace FILEID with the id of the zip file in my google drive, provided after contacting for data access.\

2. Post-hoc
3. Interpretable


