#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include <immintrin.h>  // Include header for SIMD intrinsics
#include <fstream>


// loading cuda: module load cuda11/11.0
// Compile as:
// nvcc -o lbm lbm.cu -std=c++11


using namespace std;
// define gpu and cpu kernels 
// baseline
__global__ void compute_disparity_v0(int width, int height, int patch_size, int search_range, const unsigned char* left_gray, const unsigned char* right_gray, unsigned char* disparity) {



    int min_sad = INT_MAX;
    int best_offset = 0;

    // Compute the valid range of disparities for the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= patch_size/2 && x < width - patch_size/2 && y >= patch_size/2 && y < height - patch_size/2) {

			int min_disp_range = max(0, x - search_range);
    	int max_disp_range = min(width - 1, x + search_range);
		  // Iterate over all possible disparities
		  for (int offset = min_disp_range - x; offset <= max_disp_range - x; offset++) {
		      // Compute SAD between left and right block
		      int sad = 0;
		      for (int i = -patch_size/2; i <= patch_size/2; i++) {
		          for (int j = -patch_size/2; j <= patch_size/2; j++) {

				            int px1 = left_gray[(y + i) * width + (x + j)];
				            int px2 = right_gray[(y + i) * width + (x + offset + j)];
				            sad += abs(px1 - px2);
								
		          }
		      }

		      // Update best disparity if current SAD is lower
		      if (sad < min_sad) {
		          min_sad = sad;
		          best_offset = offset;
		      }
		  }

		  // Store best disparity
		  disparity[(y * width) + x] = abs(best_offset); 
		}
}

// shared memory for left image
__global__ void compute_disparity_v1(int width, int height, int patch_size, int search_range, const unsigned char* left_gray, const unsigned char* right_gray, unsigned char* disparity) {

    __shared__ unsigned char left_shared[64][64]; // padding shared memory enough to support getting SAD neighbors for the left image tiles  

    int min_sad = INT_MAX;
    int best_offset = 0;

    // Compute the valid range of disparities for the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load left image tile into shared memory
    for (int i = -patch_size/2; i <= patch_size/2; i++) {
        for (int j = -patch_size/2; j <= patch_size/2; j++) {
					// Compute the clamped x and y values to ensure that the indices used to access left_gray
					// are within the valid range of indices for the left image array
            int clamped_x = min(max(0, x + j), width - 1);
            int clamped_y = min(max(0, y + i), height - 1);
            left_shared[threadIdx.y + i + patch_size/2][threadIdx.x + j + patch_size/2] = left_gray[y * width + x];

        }
    }

    //__syncthreads();

    if (x >= patch_size/2 && x < width - patch_size/2 && y >= patch_size/2 && y < height - patch_size/2) {
        int min_disp_range = max(0, x - search_range);
        int max_disp_range = min(width - 1, x + search_range);

        // Iterate over all possible disparities
        for (int offset = min_disp_range - x; offset <= max_disp_range - x; offset++) {
            // Compute SAD between left and right block
            int sad = 0;
            for (int i = -patch_size/2; i <= patch_size/2; i++) {
                for (int j = -patch_size/2; j <= patch_size/2; j++) {
                    int px1 = left_shared[threadIdx.y + i + patch_size/2][threadIdx.x + j + patch_size/2];
                    int px2 = right_gray[(y + i) * width + (x + offset + j)];
                    sad += abs(px1 - px2);
                }
            }
						// __syncthreads();
            // Update best disparity if current SAD is lower
            if (sad < min_sad) {
                min_sad = sad;
                best_offset = offset;
            }
        }

        // Store best disparity
        disparity[(y * width) + x] = abs(best_offset);
    }
}


// early termination
__global__ void compute_disparity_v2(int width, int height, int patch_size, int search_range, const unsigned char* left_gray, const unsigned char* right_gray, unsigned char* disparity) {

    int min_sad = INT_MAX;
    int best_offset = 0;

    // Compute the valid range of disparities for the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= patch_size/2 && x < width - patch_size/2 && y >= patch_size/2 && y < height - patch_size/2) {

		int min_disp_range = max(0, x - search_range);
		int max_disp_range = min(width - 1, x + search_range);
		// Iterate over all possible disparities
		int offset = min_disp_range - x; 	
		exit: while(offset <= max_disp_range - x){
	  		// Compute SAD between left and right block
		  	int sad = 0;
		  	for (int i = -patch_size/2; i <= patch_size/2; i++) {
				for (int j = -patch_size/2; j <= patch_size/2; j++) {

				    int px1 = left_gray[(y + i) * width + (x + j)];
				    int px2 = right_gray[(y + i) * width + (x + offset + j)];
				    sad += abs(px1 - px2);
								
		      	}
				if(sad >= min_sad){
					offset++;
					goto exit;
				}
			}

		  	// Update best disparity if current SAD is lower
		  	if (sad < min_sad) {
			  	min_sad = sad;
			  	best_offset = offset;
		  	}
			offset++;
	  	}

  		// Store best disparity
  		disparity[(y * width) + x] = abs(best_offset); 
	}
}

// shared memory for left image + early termination

__global__ void compute_disparity_v3(int width, int height, int patch_size, int search_range, const unsigned char* left_gray, const unsigned char* right_gray, unsigned char* disparity) {

    __shared__ unsigned char left_shared[64][64]; // padding shared memory enough to support getting SAD neighbors for the left image tiles  

    int min_sad = INT_MAX;
    int best_offset = 0;

    // Compute the valid range of disparities for the current pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load left image tile into shared memory
    for (int i = -patch_size/2; i <= patch_size/2; i++) {
        for (int j = -patch_size/2; j <= patch_size/2; j++) {
			// Compute the clamped x and y values to ensure that the indices used to access left_gray
			// are within the valid range of indices for the left image array
            int clamped_x = min(max(0, x + j), width - 1);
            int clamped_y = min(max(0, y + i), height - 1);
            left_shared[threadIdx.y + i + patch_size/2][threadIdx.x + j + patch_size/2] = left_gray[clamped_y * width + clamped_x];
        }
    }

    __syncthreads();

    if (x >= patch_size/2 && x < width - patch_size/2 && y >= patch_size/2 && y < height - patch_size/2) {
        int min_disp_range = max(0, x - search_range);
        int max_disp_range = min(width - 1, x + search_range);

        // Iterate over all possible disparities
		int offset = min_disp_range - x; 	
        exit: while(offset <= max_disp_range - x){
        // Compute SAD between left and right block
        int sad = 0;
        for (int i = -patch_size/2; i <= patch_size/2; i++) {
            for (int j = -patch_size/2; j <= patch_size/2; j++) {
                int px1 = left_shared[threadIdx.y + i + patch_size/2][threadIdx.x + j + patch_size/2];
                int px2 = right_gray[(y + i) * width + (x + offset + j)];
                sad += abs(px1 - px2);
            }
			if(sad >= min_sad){
				offset++;
				goto exit;
			}

        }
		//__syncthreads();
					
        // Update best disparity if current SAD is lower
        if (sad < min_sad) {
            min_sad = sad;
            best_offset = offset;
        }
					offset++;
    }

        // Store best disparity
        disparity[(y * width) + x] = abs(best_offset);
    }
}


// tiled based workload mapping (will not be included in the presentation/report --> hard to explain)

__global__ void compute_disparity_v4(int width, int height, int patch_size, int search_range, const unsigned char* left_gray, const unsigned char* right_gray, unsigned char* disparity) {
    int tile_width = blockDim.x;
    int tile_height = blockDim.y;
    int tile_x = blockIdx.x * tile_width;
    int tile_y = blockIdx.y * tile_height;

    int x_start = tile_x + threadIdx.x;
    int y_start = tile_y + threadIdx.y;

    for (int y = y_start; y < tile_y + tile_height && y < height - patch_size/2 && y >=patch_size/2; y += blockDim.y) {
        for (int x = x_start; x < tile_x + tile_width && x < width - patch_size/2 && x >=patch_size/2; x += blockDim.x) {

            int min_disp_range = max(0, x - search_range);
            int max_disp_range = min(width - 1, x + search_range);

            int min_sad = INT_MAX;
            int best_offset = 0;
			int offset = min_disp_range - x; 	
            exit: while(offset <= max_disp_range - x){
                int sad = 0;
                for (int i = -patch_size/2; i <= patch_size/2; i++) {
                    for (int j = -patch_size/2; j <= patch_size/2; j++) {
                        int px1 = left_gray[(y + i) * width + (x + j)];
                        int px2 = right_gray[(y + i) * width + (x + offset + j)];
                        sad += abs(px1 - px2);

                    }
					if(min_sad <= sad){
							offset++;
							goto exit;
					}
                }

                if (sad < min_sad) {
                    min_sad = sad;
                    best_offset = offset;
                }
				offset++;
            }

            disparity[y * width + x] = abs(best_offset);
        }
    }
}

// Function to convert RGB image to grayscale
void rgb2gray(unsigned char* rgb_img, int img_width, int img_height, unsigned char* gray_img)
{
    for (int i = 0; i < img_width * img_height; i++) {
        int r = rgb_img[3 * i];
        int g = rgb_img[3 * i + 1];
        int b = rgb_img[3 * i + 2];
        gray_img[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

// Function to normalize image for visualization
void normalize_image(unsigned char* img, int img_width, int img_height)
{
  	vector<unsigned char> img_vec(img, img + img_width * img_height);
    auto minmax = minmax_element(img_vec.begin(), img_vec.end());
    int min_val = *minmax.first;
    int max_val = *minmax.second;
    for (int i = 0; i < img_width * img_height; i++) {
        img[i] = (unsigned char)(255.0 * (img[i] - min_val) / (max_val - min_val));
    }
}
