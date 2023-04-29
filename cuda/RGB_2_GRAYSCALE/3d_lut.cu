#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>

__constant__ float factors[16 * 16 * 16];

  //3D thread block organization for the image
__global__ void rgb_to_gry_img(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height, int depth, int lut_dim, unsigned char *gry){
  
     int d=blockIdx.x*blockDim.x+threadIdx.x;    
     int w=blockIdx.y*blockDim.y+threadIdx.y;
     int h=blockIdx.z*blockDim.z+threadIdx.z;

     if(w <width && h<height && d<depth){    //boundary check
     	int index= (d*height+h)*width+w;   //indexing in 1D for the 3D input array
   
       	unsigned char red=r[index];           //getting the pixel for red
       	unsigned char green=g[index];      //getting the pixel for green
      	unsigned char blue=b[index];  //getting the pixel for blue
   
      	gry[index]= (unsigned char)factors[((red * lut_dim + blue) * lut_dim) + green];     
   }
}

__host__ void transpose(unsigned char *in, unsigned char *out, int width, int height){	
	
	int depth = 3;
    
	for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            for(int i = 0; i < depth; i++){
				out[i * height * width + (row * width) + col] = in[(row * width + col) * depth + i];
            }
        }
    }
}

int main(int argc, char *argv[]) {

	int lut_dim = 16;
	float *h_lut_ar = (float*) malloc(lut_dim * lut_dim * lut_dim * sizeof(float));
    int r,g,b;
    float gray;
    
    for(r=0;r<256;r+=16){
       	for(g=0;g<256;g+=16){
           	for(b=0;b<256;b+=16){
            	gray = 0.21*r + 0.71*g + 0.07*b;			     	
				//printf("%d %d %d %f\n", r, g, b, gray);				
				h_lut_ar[(((r / lut_dim) * lut_dim) + (g / lut_dim)) * lut_dim + (b / lut_dim)] = gray;   		
			}
   		}
  	}
	

	float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    //int element;
    int imgheight;
    int imgwidth;
    int imgdepth;

    const char *right_image_path = argv[1];
    const char *right_output_image = "output_3D_lut.png";
    unsigned char *h_image = stbi_load(right_image_path, &imgwidth, &imgheight, &imgdepth, 3);
	unsigned char *h_gray=(unsigned char*)malloc(imgwidth * imgheight * sizeof(unsigned char));
    //split image into separate channels 
    
    unsigned char *h_r=(unsigned char*)malloc(imgwidth * imgheight * sizeof(unsigned char));
    unsigned char *h_g=(unsigned char*)malloc(imgwidth * imgheight * sizeof(unsigned char));
    unsigned char *h_b=(unsigned char*)malloc(imgwidth * imgheight * sizeof(unsigned char));

	unsigned char *transposed=(unsigned char*)malloc((imgwidth * imgheight * imgdepth) * sizeof(unsigned char));	
	transpose(h_image, transposed, imgwidth, imgheight);

	for(int i=0;i<imgdepth;i++){
		for(int j=0; j<imgheight;j++){
		   for(int k=0;k<imgwidth;k++) {
               h_r[j * imgwidth + k] = transposed[(i * imgdepth + j) * imgwidth + k] / lut_dim;
               h_g[j * imgwidth + k] = transposed[(i * imgdepth + j) * imgwidth + k + 1] / lut_dim;
               h_b[j * imgwidth + k] = transposed[(i * imgdepth + j) * imgwidth + k + 2] / lut_dim;
      	   }
        }
    }

	unsigned char *d_in;
	unsigned char *d_r;
    unsigned char *d_g;
    unsigned char *d_b;
	unsigned char *d_out;
	
	cudaMalloc((void **)&d_in,(imgwidth * imgheight * imgdepth) * sizeof(unsigned char));
    cudaMalloc((void **)&d_r,(imgwidth * imgheight) * sizeof(unsigned char));
    cudaMalloc((void **)&d_g,(imgwidth * imgheight) * sizeof(unsigned char));
    cudaMalloc((void **)&d_b,(imgwidth * imgheight) * sizeof(unsigned char));
	cudaMalloc((void **)&d_out,(imgwidth * imgheight) * sizeof(unsigned char));

	cudaMemcpy(d_in, transposed, (imgwidth * imgheight * imgdepth) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, (imgwidth * imgheight) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, (imgwidth * imgheight) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (imgwidth * imgheight) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(factors, h_lut_ar, lut_dim*lut_dim*lut_dim*sizeof(float));

	const dim3 blocksize(3,16,16); //declaring the number of threads
    const dim3 gridsize((imgdepth-1)/3+1, (imgwidth-1)/16+1,(imgheight-1)/16+1); //declaring the number of blocks

	for (int j=0; j<10 ; j++){
	cudaEventRecord(start_time, 0);
    	rgb_to_gry_img<<<gridsize,blocksize>>>(d_r, d_g, d_b, imgwidth, imgheight, imgdepth, lut_dim, d_out);
	}
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
    printf("Average execution time for LUT (3D): \t%f\n",execution_time);
  
    cudaMemcpy(h_gray, d_out, imgwidth * imgheight, cudaMemcpyDeviceToHost);

    //saving the output right image
    stbi_write_png(right_output_image, imgwidth, imgheight, 1, h_gray, imgwidth);
    
    //freeing the device memory
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_out);
    
    //freeing the host memory
    free(h_gray);
    free(h_r);
    free(h_g);
    free(h_b);
    stbi_image_free(h_image);

    /*
    
    float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    //allocationg memory on the GPU for the right image
    cudaMalloc((void **)&d_r,imgwidth * imgheight * sizeof(unsigned char));
    cudaMalloc((void **)&d_g,imgwidth * imgheight * sizeof(unsigned char));
    cudaMalloc((void **)&d_b,imgwidth * imgheight * sizeof(unsigned char));
    cudaMalloc((void **)&d_gray, imgwidth * imgheight * sizeof(unsigned char));
    
    //copying image data from Host to device for the right image
    cudaMemcpy(d_r, h_r, imgwidth*imgheight*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, imgwidth*imgheight*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, imgwidth*imgheight*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    const dim3 blocksize(32,32,1); //declaring the number of threads
    const dim3 gridsize((imgwidth-1)/32+1,(imgheight-1)/32+1,(imgdepth-1)/1+1); //declaring the number of blocks
    
    cudaEventRecord(start_time, 0);
    //calling the kernel
    //for (int j=0; j<10 ; j++){
    rgb_to_gry_img<<<gridsize,blocksize>>>(d_r, d_g, d_b, imgwidth, imgheight, imgdepth, d_gray);
    //}
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    //execution_time /=10.0f;
    printf("Total execution time (ms) for 3D Color To Grayscale conversion %f\n",execution_time);
  
    cudaMemcpy(h_gray, d_gray, imgwidth * imgheight, cudaMemcpyDeviceToHost);

    //saving the output right image
    stbi_write_png(right_output_image, imgwidth, imgheight, 1, h_gray, imgwidth);
    
    //freeing the device memory
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_gray);
    
    //freeing the host memory
    free(h_gray);
    free(h_r);
    free(h_g);
    free(h_b);
    stbi_image_free(h_image);
*/
  return 0;
}
