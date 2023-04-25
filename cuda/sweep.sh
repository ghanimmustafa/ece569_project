#!/bin/bash

# Define the input image directory
IMG_DIR="./img/Motorcycle/"
DISPARITY_RANGE=("270")

# Define the gpu_block configurations
GPU_BLOCKS=("1" "2" "4" "8" "16" "32")

# Define the SAD_block configurations
SAD_BLOCKS=("3" "5" "7" "9" "11")

# Loop over the image names, gpu_block configurations, and SAD_block configurations

for GPU_BLOCK in "${GPU_BLOCKS[@]}"
do
    for SAD_BLOCK in "${SAD_BLOCKS[@]}"
    do
        # Run the lbm program with the current configuration
        ./lbm $GPU_BLOCK $SAD_BLOCK $DISPARITY_RANGE $IMG_DIR
    done
done


