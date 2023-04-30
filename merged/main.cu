#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
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
#include "lbm.cu"
#include "color_conversion.cu"

int main(int argc, char** argv)
{
	// Check command line arguments
	if (argc != 7) {
			cerr << "Usage: " << argv[0] << " <kernel_version> <gpu_patch_size> <cost block size> <search range> <image directory> <color_conversion_version>" << endl;
			return 1;
	}

    // Parse block size and search range from command line arguments
	int kernel_version = atoi(argv[1]);
	int gpu_block_dim = atoi(argv[2]);
    int patch_size = atoi(argv[3]);
    int search_range = atoi(argv[4]);
	string image_dir = argv[5];
	int color_conversion_version = atoi(argv[6]);
   	string left_path = image_dir + "/view5.png";
   	string right_path = image_dir + "/view1.png";
	typedef void (*kernel_function_t)(int, int, int, int, const unsigned char*, const unsigned char*, unsigned char*);

	kernel_function_t kernel_functions[] = {
			compute_disparity_v0, // baseline
			compute_disparity_v1, // shared memory only
			compute_disparity_v2, // early termination only
			compute_disparity_v3, // shared memory + early termination
			compute_disparity_v4
	};

	kernel_function_t kernel_function = kernel_functions[kernel_version];
    // Check that block size and search range are valid
    if (patch_size <= 0 || patch_size % 2 == 0 || search_range <= 0) {
        cerr << "Error: invalid block size or search range" << endl;
        return 1;
    }


    // Read input images
    int left_width, left_height, left_channels;
    unsigned char* left_data = stbi_load(left_path.c_str(), &left_width, &left_height, &left_channels, 3);
    if (!left_data) {
        cerr << "Error: could not read the left input image 'view5.png'" << endl;
        return 1;
    }

    int right_width, right_height, right_channels;
    unsigned char* right_data = stbi_load(right_path.c_str(), &right_width, &right_height, &right_channels, 3);
    if (!right_data) {
        cerr << "Error: could not read the right image 'view1.png'" << endl;
        return 1;
    }

    // Convert input images to grayscale

    unsigned char* left_gray_data = new unsigned char[left_width * left_height];
    //rgb2gray(left_data, left_width, left_height, left_gray_data);
    run_color_conversion(left_data, left_gray_data, left_width, left_height, color_conversion_version);

    unsigned char* right_gray_data = new unsigned char[right_width * right_height];
    //rgb2gray(right_data, right_width, right_height, right_gray_data);
	run_color_conversion(right_data, right_gray_data, right_width, right_height, color_conversion_version);

	// Create output_images directory if it doesn't exist
	string output_dir = image_dir + "output_images/";
	DIR* dir = opendir(output_dir.c_str());
	if (dir) {
		closedir(dir);
	} else {
		mkdir(output_dir.c_str(), 0777);
	}  

	string output_path = output_dir + "/left_gray.png";
	// Write grayscale images to disk
	if (!stbi_write_png(output_path.c_str(), left_width, left_height, 1, left_gray_data, left_width)) {
		cerr << "Error: could not write output image 'left_gray.png'" << endl;
		return 1;
	}

  	output_path = output_dir + "/right_gray.png";	

	if (!stbi_write_png(output_path.c_str(), right_width, right_height, 1, right_gray_data, right_width)) {
		cerr << "Error: could not write output image 'right_gray.png'" << endl;
		return 1;
	}


	// Allocate memory for output disparity map
	int disp_width = left_width;
	int disp_height = left_height;
	unsigned char* disparity = new unsigned char[disp_width * disp_height];
	auto start_time = std::chrono::system_clock::now();   
	// Declare and allocate device memory
	unsigned char *d_left_gray, *d_right_gray, *d_disparity;
	cudaMalloc((void **)&d_left_gray, left_width * left_height * sizeof(unsigned char));
	cudaMalloc((void **)&d_right_gray, right_width * right_height * sizeof(unsigned char));
	cudaMalloc((void **)&d_disparity, disp_width * disp_height * sizeof(unsigned char));

	// Copy input data to device
	cudaMemcpy(d_left_gray, left_gray_data, left_width * left_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_gray, right_gray_data, right_width * right_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// Set up kernel grid and block size
	//int gpu_block_dim = 2;
	dim3 block(gpu_block_dim,gpu_block_dim, 1);
	dim3 grid(ceil(disp_width + gpu_block_dim - 1) / gpu_block_dim, ceil(disp_height + gpu_block_dim - 1) / gpu_block_dim, 1);

	float elapsed_time = 0.0f;
	for (int i = 0; i < 10; i++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		kernel_function<<<grid, block>>>(disp_width, disp_height, patch_size, search_range, d_left_gray, d_right_gray, d_disparity);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float time_i = 0.0f;
		cudaMemcpy(disparity, d_disparity, disp_width * disp_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaEventElapsedTime(&time_i, start, stop);
		elapsed_time += time_i;
	}
	float avg_time = elapsed_time / 10.0f;
    // Write the input configurations and elapsed time to CSV file
    ofstream outfile;
    outfile.open("results.csv", ios_base::app); // open the file in append mode
    outfile << image_dir << "," << kernel_version << "," << gpu_block_dim << "," << patch_size << "," << search_range << "," << avg_time << endl;
    outfile.close();
    
    cout << "Average disparity estimation kernel execution time: " << avg_time << "ms" << endl;

	auto end_time = std::chrono::system_clock::now();
	auto program_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	cout << "Total execution time: " << program_time << "ms" << endl;
	// Normalize output image for visualization
	normalize_image(disparity, disp_width, disp_height);

	output_path = output_dir + "/disparity_gpu_" + to_string(patch_size) + "_" + to_string(search_range)  + "_.png";

	// Write output disparity map to file
	stbi_write_png(output_path.c_str(), disp_width, disp_height, 1, disparity, disp_width);

	// Free device and host memory
	cudaFree(d_left_gray);
	cudaFree(d_right_gray);
	cudaFree(d_disparity);
	delete[] left_data;
	delete[] right_data;
	delete[] left_gray_data;
	delete[] right_gray_data;
	delete[] disparity;
	
	return 0;    
	
}
