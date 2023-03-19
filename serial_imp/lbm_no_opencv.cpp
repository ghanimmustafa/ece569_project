#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
// Compile as:
// g++ lbm_no_opencv.cpp -o lbm_no_opencv `pkg-config --cflags --libs opencv4`


using namespace std;
// Function to compute sum of absolute differences between two image blocks (currently implemented in the main directly)
/*int sad(const unsigned char* img1, const unsigned char* img2, int img_width, int x1, int y1, int x2, int y2)
{
    int sum = 0;
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int px1 = img1[(y1 + i) * img_width + (x1 + j)];
            int px2 = img2[(y2 + i) * img_width + (x2 + j)];
            sum += abs(px1 - px2);
        }
    }
    return sum;
}*/

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
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <block size> <search range>" << endl;
        return 1;
    }

    // Parse block size and search range from command line arguments
    int block_size = atoi(argv[1]);
    int search_range = atoi(argv[2]);

    // Check that block size and search range are valid
    if (block_size <= 0 || block_size % 2 == 0 || search_range <= 0) {
        cerr << "Error: invalid block size or search range" << endl;
        return 1;
    }

    // Read input images
    int left_width, left_height, left_channels;
    unsigned char* left_data = stbi_load("view5.png", &left_width, &left_height, &left_channels, 3);
    if (!left_data) {
        cerr << "Error: could not read the left input image 'view5.png'" << endl;
        return 1;
    }

    int right_width, right_height, right_channels;
    unsigned char* right_data = stbi_load("view1.png", &right_width, &right_height, &right_channels, 3);
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
		// Write grayscale images to disk
		if (!stbi_write_png("left_gray.png", left_width, left_height, 1, left_gray_data, left_width)) {
				cerr << "Error: could not write output image 'left_gray.png'" << endl;
				return 1;
		}

		if (!stbi_write_png("right_gray.png", right_width, right_height, 1, right_gray_data, right_width)) {
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

	// Write output disparity map to file
	stbi_write_png("disparity.png", disp_width, disp_height, 1, disp_data, disp_width);

	// Free memory
	delete[] left_data;
	delete[] right_data;
	delete[] left_gray_data;
	delete[] right_gray_data;
	delete[] disp_data;

	return 0;    
	
}
		  

