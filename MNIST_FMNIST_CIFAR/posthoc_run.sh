#!/bin/bash

echo "CIFAR"
python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.05" --validation "with_test"  --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.05"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.10" --validation "with_test"  --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.10"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.25" --validation "with_test"  --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.25"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.50" --validation "with_test"  --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.50"

echo "Experiments for CIFAR DONE!!!!"


echo "FMNIST"
python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.05" --validation "with_test" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.05"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.10" --validation "with_test" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.10"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.25" --validation "with_test" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.25"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.50" --validation "with_test" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --load_bb_model "false" 
echo " --num_patches 0.50"

echo "Experiments for FMNIST DONE!!!!"


echo "MNIST"
python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.05" --validation "with_test"  --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --load_bb_model "true" 
echo " --num_patches 0.05"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.10" --validation "with_test"  --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --load_bb_model "true" 
echo " --num_patches 0.10"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.25" --validation "with_test"  --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --load_bb_model "true" 
echo " --num_patches 0.25"

python ./MNIST_FMNIST_CIFAR/post_hoc.py --num_patches "0.50" --validation "with_test"  --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --load_bb_model "true" 
echo " --num_patches 0.50"

echo "Experiments for MNIST DONE!!!!"