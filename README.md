# ATtention for CAusal eXPLAINation (CAT-XPLAIN)
This project aims at providing causal explaination capability for existing Neural Network Architectures. It utilizes attention mechanism of Transformers to attend and hence identify the most important regions of input that have highest Causal significance to the output. This project is motivated from a paper titled ["Instance-wise Causal Feature Selection for Model Interpretation", 2021 by Pranoy et. al.](https://openaccess.thecvf.com/content/CVPR2021W/CiV/papers/Panda_Instance-Wise_Causal_Feature_Selection_for_Model_Interpretation_CVPRW_2021_paper.pdf) 

Their paper proposes to build a model agnostic post-hoc explainer model that is able to identify the most causally significant regions in the input space (features selection) of each instance. Unlike the post-hoc explanation approach, we propose to minor modification on the existing Transformer architecture so that the model is able to inherently identify the regions with highest causal strength while performing the task they are designed for. This leads to development of Transformers with causal explaination capability without the need of additional post-hoc explainer.


### Steps (for MNIST and FMNIST and CIFAR)

1. `git clone git@github.com:mvrl/CAT-XPLAIN.git`
2. `cd CAT-XPLAIN`
3. Create a virtual environment for the project.
    `conda env create -f environment.yml`
4.  `conda activate CAT-XPLAIN`
5. Example: Run the post-hoc experiment with 25% unmasked patches for MNIST or FMNIST or CIFAR datasets

    `python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.25" --validation "with_test"  --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512`
    
    `python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.25" --validation "with_test" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512`

    `python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.25" --validation "with_test"  --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512`

6. Example: Run the Interpretable transformer with 25% unmasked patches for MNIST or FMNIST or CIFAR datasets with loss_weight 0.90

    `python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --validation  "with_test" --num_patches "0.25"`

    `python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --validation  "with_test" --num_patches "0.25"`

    `python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --validation  "with_test" --num_patches "0.25"`

<!-- ### Steps (for IMDB dataset)
1. Run the post-hoc experiment 
    `python ./IMDB/post_hoc_imdb.py --num_words 0.25 --validation "with_test" --bb_model_type "transformer" --sel_model_type "transformer"`


2. Run the Interpretable transformer for IMDB dataset
    `python ./IMDB/interpretable_transformer_imdb.py --num_words 0.25 --validation "with_test" --loss_weight 0.9` -->


<!-- ### Steps (for IMDB_sentence experiment)
1. Download dataset 
    `wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P path_to_storage_folder`
2. Unzip the file
    `tar -xvf path_to_storage_folder/aclImdb_v1.tar.gz -C path_to_storage_folder`
3. Prep data: Merge train-test, split train/val/test ratio 0.70:0.10:0.20; sentence counts: 10 to 50. 
    `python ./IMDB_sentence/data_prep.py`

4. Download one hugging face sentence transformer model for embedding sentences.
    Information about all options can be seen in [Hugging face pretained models](https://www.sbert.net/docs/pretrained_models.html)
    `git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 path_to_storage_folder`

5. Run the post-hoc experiment 
    `python ./IMDB_sentence/post_hoc_imdb.py --num_sents 0.25 --validation "with_test" --bb_model_type "transformer" --sel_model_type "transformer"`

6. Run the Interpretable transformer for IMDB dataset

    `python ./IMDB_sentence/interpretable_transformer_imdb.py --num_sents 0.25 --validation "with_test" --loss_weight 0.9` -->



<!-- ### Steps (for ADNI MRI dataset)

1. Download our preprocessed ADNI data and cv splits using the FILEIDS provided after access request at subash.khanal33@gmail.com
    
    `gdown -O "storage_path/ADNI.zip" --id "1C7y9nviFU4HCtthKOPBLjvvxBhLKI511"`

    `gdown -O ./MRI/cv_paths.zip --id "11pPZTKnu9E_ZqeCL_7LnEhC5KNq-J1Qr"`


2. Unzip the zipped files

    `unzip storage_path/ADNI.zip -d storage_path`

    `unzip ./MRI/cv_paths.zip -d ./MRI`

3. Post-hoc experiment for MRI data

    `python ./MRI/post_hoc_mri.py --num_patches 0.25 --validation "with_test"`

4. Interpretable ViT experiment for MRI data

    `python ./MRI/interpretable_transformer_mri.py --num_patches 0.25 --validation "with_test" --loss_weight 0.9` -->



