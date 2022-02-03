# ATtention for CAusal eXPLAINation (CAT-XPLAIN)
This project aims at providing causal explaination capability for existing Neural Network Architectures. It utilizes attention mechanism of Transformers to attend and hence identify the most important regions of input that have highest Causal significance to the output. This project is motivated from a paper titled ["Instance-wise Causal Feature Selection for Model Interpretation", 2021 by Pranoy et. al.](https://openaccess.thecvf.com/content/CVPR2021W/CiV/papers/Panda_Instance-Wise_Causal_Feature_Selection_for_Model_Interpretation_CVPRW_2021_paper.pdf) 

Their paper proposes to build a model agnostic post-hoc explainer model that is able to identify the most causally significant regions in the input space (features selection) of each instance. Unlike the post-hoc explanation approach, we propose to modify the existing Transformer architecture so that the model able to inherently identify the regions with highest causal strength while performing the task they are designed for. This leads to development of Transformers with causal explaination capability without the need of additional post-hoc explainer.


### Steps (for MNIST and FMNIST)

1. `https://github.com/Subash33/CAT-XPLAIN`
2. `cd CAT-XPLAIN`
3. Create a virtual environment for the project.\
    `conda env create -f environment.yml`
4.  `conda activate CAT-XPLAIN`
5. Run the post-hoc experiment for MNIST or FMNIST datasets

    `python ./MNIST_FMNIST/post_hoc.py --num_patches 0.25 --validation "with_test" --bb_model_type "ViT" --sel_model_type "ViT" --dataset_name "mnist"`

    `python ./MNIST_FMNIST/post_hoc.py --num_patches 0.25 --validation "with_test" --bb_model_type "ViT" --sel_model_type "ViT" --dataset_name "fmnist"`

6. Run the Interpretable transformer for MNIST or FMNIST datasets

    `python ./MNIST_FMNIST/interpretable_transformer.py --num_patches 0.25 --validation "with_test" --loss_weight 0.9 --dataset_name "mnist"`

    `python ./MNIST_FMNIST/interpretable_transformer.py --num_patches 0.25 --validation "with_test" --loss_weight 0.9 --dataset_name "fmnist"`

### Steps (for IMDB dataset)
1. Run the post-hoc experiment 
    `python ./IMDB/post_hoc_imdb.py --num_patches 0.25 --validation "with_test" --bb_model_type "transformer" --sel_model_type "transformer"`


2. Run the Interpretable transformer for MNIST or FMNIST datasets

    `python ./IMDB/interpretable_transformer_imdb.py --num_patches 0.25 --validation "with_test" --loss_weight 0.9`


### Steps (for ADNI MRI dataset)

1. Download our preprocessed ADNI data and cv splits using the FILEIDS provided after access request at subash.khanal33@gmail.com
    
    `gdown -O "storage_path/ADNI.zip" --id "1C7y9nviFU4HCtthKOPBLjvvxBhLKI511"`

    `gdown -O ./MRI/cv_paths.zip --id "11pPZTKnu9E_ZqeCL_7LnEhC5KNq-J1Qr"`


2. Unzip the zipped files

    `unzip storage_path/ADNI.zip -d storage_path`

    `unzip ./MRI/cv_paths.zip -d ./MRI`

3. Post-hoc experiment for MRI data

    `python ./MRI/post_hoc_mri.py --num_patches 0.25 --validation "with_test"`

4. Interpretable ViT experiment for MRI data

    `python ./MRI/interpretable_transformer_mri.py --num_patches 0.25 --validation "with_test" --loss_weight 0.9`



