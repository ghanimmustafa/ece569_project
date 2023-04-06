#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#define RANK_SIZE 25 

// Header files for image read/write
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void grayscale_conversion(uint8_t *img, int height, int width, int depth, uint8_t *return_array){
    double r, g, b, gray;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
                r = (double)img[(i * width + j) * depth + 0];
                g = (double)img[(i * width + j) * depth + 1];
                b = (double)img[(i * width + j) * depth + 2];

                r *= 0.299;
                g *= 0.587;
                b *= 0.114;

                gray = (r + g + b);

				if(gray > 255)printf("%f\n", gray);
                return_array[i * width + j] = (uint8_t)gray;
        }
    }
}

void rankTransform(uint8_t* img, int rankSize, int height, int width, uint8_t* output_array){



	int borders = std::floor(rankSize / 2);
	int center = borders + 1;
	int index = 0;

	uint8_t *f = (uint8_t*) malloc (sizeof(uint8_t) * (rankSize * rankSize));


	for(int iy = 1 + borders; iy < height - borders; iy++){
		for(int ix = 1 + borders; ix < width - borders; ix++){
			//printf("%d %d %d %d\n", iy-borders,iy+borders,ix-borders,ix+borders);
			for(int ii = iy - borders; ii < iy + borders; ii++){
				for(int jj = ix - borders; jj < ix + borders; jj++){
					//printf("%d %d %d\n", ii, jj, ii * (width) + jj);
					f[index++] = img[ii * width + jj];
					//printf("%d\n", index++);
				}
			}
			int iix = ix - borders;
			int iiy = iy - borders;
			for(int ii = 0; ii < rankSize; ii++){
				for(int jj = 0; jj < rankSize; jj++){
					if(f[center * rankSize + center] > f[ii * rankSize +jj]) f[ii * rankSize +jj] = 1;
					else f[ii * rankSize +jj] = 0;
				}
			}

			uint8_t *f_ = (uint8_t*) malloc (sizeof(uint8_t) * (rankSize));
			uint8_t sum = 0;
			int index_ = 0;

			for(int ii = 0; ii < rankSize; ii++){
				for(int jj = 0; jj < rankSize; jj++){
					sum += f[ii * rankSize + jj];
				}
			f_[index_++] = sum;
			sum = 0;
			}

			uint8_t out = 0;
			for(int ii = 0; ii < rankSize; ii++){
				out += f_[ii];
			}

			output_array[iiy * width + iix] = out;
			out = 0;
			index = 0;
		}
	}

}

int main(){

	int height;// = element;
    //f >> element;
    int width;// = element;
    //f >> element;
    int depth;// = element;

    char left_imagepath[100] = "./Room/left.ppm";
    uint8_t* left_rgb_imag = stbi_load(left_imagepath, &width, &height, &depth, 3);
	char right_imagepath[100] = "./Room/right.ppm";
    uint8_t* right_rgb_imag = stbi_load(right_imagepath, &width, &height, &depth, 3);

    if (left_rgb_imag == NULL){
    std::cout << "Failed to open image " << left_imagepath << std::endl;
        return -1;
    }
	if (right_rgb_imag == NULL){
    std::cout << "Failed to open image " << right_imagepath << std::endl;
        return -1;
    }
    depth = 3;
    int size = height * width;

	
	/*double *left_img = (double*) malloc (sizeof(double) * (size * 3));
	double *right_img = (double*) malloc (sizeof(double) * (size * 3));*/
	uint8_t *grayscale_left_img = (uint8_t*) malloc (sizeof(uint8_t) * (size));
	uint8_t *grayscale_right_img = (uint8_t*) malloc (sizeof(uint8_t) * (size));

	/*for(int i = 0; i < size * 3; i++){
		left_img[i] = (double) left_rgb_imag[i];
		right_img[i] = (double) right_rgb_imag[i];
	}*/

	grayscale_conversion(left_rgb_imag, height, width, depth, grayscale_left_img);
	grayscale_conversion(right_rgb_imag, height, width, depth, grayscale_right_img);

	stbi_write_png("left_img_grayscale.png", width, height, 1, grayscale_left_img, width);
	stbi_write_png("right_img_grayscale.png", width, height, 1, grayscale_right_img, width);


	uint8_t *rank_left_img = (uint8_t*) malloc (sizeof(uint8_t) * (size));
	uint8_t *rank_right_img = (uint8_t*) malloc (sizeof(uint8_t) * (size));

	rankTransform(grayscale_left_img, RANK_SIZE, height, width, rank_left_img);
	rankTransform(grayscale_right_img, RANK_SIZE, height, width, rank_right_img);

	/*uint8_t *rank_left_img_debug = (uint8_t*) malloc (sizeof(uint8_t) * (size));
	uint8_t *rank_right_img_debug = (uint8_t*) malloc (sizeof(uint8_t) * (size));

	for (int i = 0; i<(height * width); i++){
        rank_left_img_debug[i] = (uint8_t)rank_left_img[i];
        rank_right_img_debug[i] = (uint8_t)rank_right_img[i];
    }*/

	stbi_write_png("left_img_rank.png", width, height, 1, rank_left_img, width);
	stbi_write_png("right_img_rank.png", width, height, 1, rank_right_img, width);

	return 0;
}
