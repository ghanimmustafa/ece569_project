#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
 
  //3D thread block organization for the image
__global__ void rgb_to_gry_img(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height, int depth, unsigned char *gry){
  
   int w=blockIdx.x*blockDim.x+threadIdx.x;    
   int h=blockIdx.y*blockDim.y+threadIdx.y;
   int d=blockIdx.z*blockDim.z+threadIdx.z;

     if(w <width && h<height && d<depth){    //boundary check
        int index= d*width*height+h*width+w;   //indexing in 1D for the 3D input array
   
       unsigned char red=r[index];           //getting the pixel for red
       unsigned char green=g[index];      //getting the pixel for green
       unsigned char blue=b[index];  //getting the pixel for blue
   
       gry[index]= 0.21*red + 0.71*green + 0.07*blue;     
   }
}

int main(int argc, char *argv[]) {
    //int element;
    int imgheight;
    int imgwidth;
    int imgdepth;

    const char *right_image_path = "view1.png";
    const char *right_output_image = "rightoutput_3D.png";
    unsigned char *h_image = stbi_load(right_image_path, &imgwidth, &imgheight, &imgdepth, 3);
    //split image into separate channels 
    
    unsigned char *h_r=(unsigned char*)malloc(imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    unsigned char *h_g=(unsigned char*)malloc(imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    unsigned char *h_b=(unsigned char*)malloc(imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    unsigned char *h_gray=(unsigned char*)malloc(imgwidth * imgheight * imgdepth* sizeof(unsigned char));

    for(int i=0;i<imgdepth;i++){
        for(int j=0; j<imgheight;j++){
           for(int k=0;k<imgwidth;k++) {
               h_r[i*imgdepth*imgheight+j*imgwidth+k] = h_image[i*imgdepth*imgheight+j*imgwidth+k];
               h_g[i*imgdepth*imgheight+j*imgwidth+k]= h_image[i*imgdepth*imgheight+j*imgwidth+k+1];
               h_b[i*imgdepth*imgheight+j*imgwidth+k]= h_image[i*imgdepth*imgheight+j*imgwidth+k+2];
          }
        }
      }
    unsigned char *d_r;
    unsigned char *d_g;
    unsigned char *d_b;
    unsigned char *d_gray;
    
    float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);   

    //allocationg memory on the GPU for the right image
    cudaMalloc((void **)&d_r,imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    cudaMalloc((void **)&d_g,imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    cudaMalloc((void **)&d_b,imgwidth * imgheight * imgdepth * sizeof(unsigned char));
    cudaMalloc((void **)&d_gray, imgwidth * imgheight * sizeof(unsigned char));
    
    //copying image data from Host to device for the right image
    cudaMemcpy(d_r, h_r, imgwidth*imgheight*imgdepth*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, imgwidth*imgheight*imgdepth*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, imgwidth*imgheight*imgdepth*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    const dim3 blocksize(32,32,1); //declaring the number of threads
    const dim3 gridsize((imgwidth-1)/32+1,(imgheight-1)/32+1,(imgdepth-1)/1+1); //declaring the number of blocks
    
    cudaEventRecord(start_time, 0);
    //calling the kernel
    for (int j=0; j<10 ; j++){
    rgb_to_gry_img<<<gridsize,blocksize>>>(d_r, d_g, d_b, imgwidth, imgheight, imgdepth, d_gray);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
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

  return 0;
}
