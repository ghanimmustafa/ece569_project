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
// loading cuda: module load cuda11/11.0
// Compile as:
// nvcc -o lbm lbm.cu -std=c++11


using namespace std;



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

int main(int argc, char** argv)
{
		// Check command line arguments
		if (argc != 4) {
				cerr << "Usage: " << argv[0] << " <block size> <search range> <image directory>" << endl;
				return 1;
		}


    // Parse block size and search range from command line arguments
    int block_size = atoi(argv[1]);
    int search_range = atoi(argv[2]);
 		string image_dir = argv[3];
   	string left_path = image_dir + "/view5.png";
   	string right_path = image_dir + "/view1.png";
    // Check that block size and search range are valid
    if (block_size <= 0 || block_size % 2 == 0 || search_range <= 0) {
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
 		auto start_time = std::chrono::system_clock::now();   
    unsigned char* left_gray_data = new unsigned char[left_width * left_height];
    rgb2gray(left_data, left_width, left_height, left_gray_data);

    unsigned char* right_gray_data = new unsigned char[right_width * right_height];
    rgb2gray(right_data, right_width, right_height, right_gray_data);
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

    // Compute disparity map
    int disp_width = left_width;
    int disp_height = left_height;
    unsigned char* disp_data = new unsigned char[disp_width * disp_height];

    // Compute maximum disparity
    int max_disparity = left_width - block_size;

		// Iterate over all pixels in left image
		for (int y = 0; y < left_height; y++) {
				for (int x = 0; x < left_width; x++) {
				    int min_sad = INT_MAX;
				    int best_offset = 0;

				    // Compute the valid range of disparities for the current pixel
				    int min_disp_range = max(0, x - search_range);
				    int max_disp_range = min(right_width - 1, x + search_range);

				    // Iterate over all possible disparities
				    for (int offset = min_disp_range - x; offset <= max_disp_range - x; offset++) {
				        // Compute SAD between left and right block
				        int sad = 0;
				        for (int i = 0; i < block_size; i++) {
				            for (int j = 0; j < block_size; j++) {
				                int px1 = left_gray_data[(y + i) * left_width + (x + j)];
				                int px2 = right_gray_data[(y + i) * right_width + (x + offset + j)];
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
				    disp_data[(y * disp_width) + x] = abs(best_offset) * (255.0 / max_disparity);
				}
		}


	// Normalize output image for visualization
	normalize_image(disp_data, disp_width, disp_height);



	auto end_time = std::chrono::system_clock::now();
	auto program_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	cout << "Execution time: " << program_time << "ms" << endl;
  output_path = output_dir + "/disparity.png"; 	
	// Write output disparity map to file
	stbi_write_png(output_path.c_str(), disp_width, disp_height, 1, disp_data, disp_width);

	// Free memory
	delete[] left_data;
	delete[] right_data;
	delete[] left_gray_data;
	delete[] right_gray_data;
	delete[] disp_data;
	

	return 0;    
	
}
		  

