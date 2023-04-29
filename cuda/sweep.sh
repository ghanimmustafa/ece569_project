#!/bin/bash

VERSIONS=("0" "1" "2" "3")
# Define the input image directory
IMG_DIR=("./img/Motorcycle")
DISPARITY_RANGE=("270")

# Define the gpu_block configurations
GPU_BLOCKS=("2" "4" "8" "16" "32")

# Define the SAD_block configurations
SAD_BLOCKS=("3" "7" "11" "15")

# Loop over the image names, gpu_block configurations, and SAD_block configurations
for SAD_BLOCK in "${SAD_BLOCKS[@]}"
do
	for GPU_BLOCK in "${GPU_BLOCKS[@]}"
	do

			for VERSION in "${VERSIONS[@]}"
				do
		      # Run the lbm program with the current configuration
		      ./lbm $VERSION $GPU_BLOCK $SAD_BLOCK $DISPARITY_RANGE $IMG_DIR
		  done
	done
done

