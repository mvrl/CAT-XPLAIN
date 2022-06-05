#!/bin/bash

echo "CIFAR"
echo "frac 0.05 loss_weight 0.90"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --validation  "with_test" --num_patches "0.05"
echo "DONE!!!!!!!-----------------------------------------------------------"


echo "CIFAR"
echo "frac 0.10 loss_weight 0.90"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --validation  "with_test" --num_patches "0.10"
echo "DONE!!!!!!!-----------------------------------------------------------"


echo "CIFAR"
echo "frac 0.25 loss_weight 0.90"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --validation  "with_test" --num_patches "0.25"
echo "DONE!!!!!!!-----------------------------------------------------------"


echo "CIFAR"
echo "frac 0.50 loss_weight 0.90"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "cifar" --dataset_class "partial" --depth 8 --dim 512 --validation  "with_test" --num_patches "0.50"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "CIFAR EXPERIMENTS ON EXP-VIT DONE!!!!"




echo "FMNIST"
echo "frac 0.05 loss_weight 0.70"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.70" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --validation  "with_test" --num_patches "0.05"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "FMNIST"
echo "frac 0.10 loss_weight 0.90"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --validation  "with_test" --num_patches "0.10"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "FMNIST"
echo "frac 0.25 loss_weight 0.50"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.50" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --validation  "with_test" --num_patches "0.25"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "FMNIST"
echo "frac 0.50 loss_weight 0.60"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.60" --dataset_name "fmnist" --dataset_class "partial" --depth 4 --dim 512 --validation  "with_test" --num_patches "0.50"
echo "DONE!!!!!!!-----------------------------------------------------------"






echo "FMNIST EXPERIMENTS ON EXP-VIT DONE!!!!"

echo "MNIST"
echo "frac 0.05 loss_weight 0.90"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.90" --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --validation  "with_test" --num_patches "0.05"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "MNIST"
echo "frac 0.10 loss_weight 0.70"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.70" --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --validation  "with_test" --num_patches "0.10"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "MNIST"
echo "frac 0.25 loss_weight 0.60"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.60" --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --validation  "with_test" --num_patches "0.25"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "MNIST"
echo "frac 0.50 loss_weight 0.70"
python ./MNIST_FMNIST_CIFAR/interpretable_transformer.py --loss_weight "0.70" --dataset_name "mnist" --dataset_class "partial" --depth 6 --dim 512 --validation  "with_test" --num_patches "0.50"
echo "DONE!!!!!!!-----------------------------------------------------------"

echo "MNIST EXPERIMENTS ON EXP-VIT DONE!!!!"