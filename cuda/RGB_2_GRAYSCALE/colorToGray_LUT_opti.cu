#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>

//@@ INSERT DEVICE CODE HERE
#define ch 3
  //2D thread organization
__global__ void rgb_to_gry_img_lut(unsigned char* rgbI, int width, int height, unsigned char* gryI, float *Lut){
 
	int w=blockIdx.x*blockDim.x+threadIdx.x;    
	int h=blockIdx.y*blockDim.y+threadIdx.y;

	if(w <width && h<height){    //boundary check
		int index= h*width+w; //indexing for the input array in 2D
    	
		int rgb_index= index*ch;    
       	unsigned char red=rgbI[rgb_index];   //getting the pixel for red in r
       	unsigned char green=rgbI[rgb_index+1]; //getting the pixel for green
       	unsigned char blue=rgbI[rgb_index+2];  //getting the pixel for blue
   	       	
		gryI[index]= (unsigned char)Lut[((red * 256 + blue) * 256) + green];
	}
}

int main(int argc, char *argv[]) {
	    
	float *h_lut_ar = (float*) malloc(256 * 256 * 256 * sizeof(float));
    int r,g,b;
    float gray;
    
    for(r=0;r<256;r++){
       	for(g=0;g<256;g++){
           	for(b=0;b<256;b++){
            	gray = 0.21*r + 0.71*g + 0.07*b;			     	
				h_lut_ar[((r * 256) + g) * 256 + b] = gray;   		
			}
   		}
  	} 
    int imgheight;
    int imgwidth;
    int imgdepth;
    float *d_lut_ar;
    const char *right_image_path = "im0.png";
    const char *right_output_image= "rightoutput_LUT.png";
    unsigned char *h_right_input_image = stbi_load(right_image_path, &imgwidth, &imgheight, &imgdepth, 3);
    unsigned char *h_right_output_image;
    unsigned char *d_right_input_image;
    unsigned char *d_right_output_image; 
  
    float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);   

    int right_image_size = imgwidth * imgheight* imgdepth * sizeof(unsigned char);
    h_right_output_image = (unsigned char*) malloc(right_image_size);
    
    //allocationg memory on the GPU for the right image
    cudaMalloc((void **)&d_right_input_image,imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    cudaMalloc((void **)&d_right_output_image, imgwidth * imgheight * sizeof(unsigned char));
    cudaMalloc((void **)&d_lut_ar, 256 * 256 * 256 * sizeof(float)); 

    //copying image data from Host to device for the right image
    cudaMemcpy(d_right_input_image, h_right_input_image, right_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut_ar,h_lut_ar,256*256*256 * sizeof(float),cudaMemcpyHostToDevice);
    
    
    int B=32;
    const dim3 blocksize(B,B,1); //declaring the number of threads
    const dim3 gridsize((imgwidth-1)/B+1,(imgheight-1)/B+1,1); //declaring the number of blocks
    
    cudaEventRecord(start_time, 0);
    //calling the kernel
    for (int j=0; j<10 ; j++){
    	rgb_to_gry_img_lut<<<gridsize,blocksize>>>(d_right_input_image, imgwidth, imgheight, d_right_output_image, d_lut_ar);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
	printf("Total execution time for LUT based Color To Grayscale conversion: (ms) %f\n",execution_time);
    

    //coping output data from Device to Host for the right image
    cudaMemcpy(h_right_output_image, d_right_output_image, imgwidth * imgheight, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lut_ar,d_lut_ar,256*256*256,cudaMemcpyDeviceToHost);
    //saving the output right image
    stbi_write_png(right_output_image, imgwidth, imgheight, 1, h_right_output_image, imgwidth);
   
    
    //freeing the device memory
    cudaFree(d_right_input_image);
    cudaFree(d_right_output_image);
    cudaFree(d_lut_ar);
  
    //freeing the host memory
    free(h_right_output_image);
    free(h_lut_ar);
    stbi_image_free(h_right_input_image);
   

  return 0;
}
