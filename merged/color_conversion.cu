#include <cuda_runtime.h>

#define ch 3

__constant__ float factors[16 * 16 * 16];

__global__ void rgb_to_gry_img_2D(unsigned char* rgbI, int width, int height, unsigned char* gryI){ 
 
   int w=blockIdx.x*blockDim.x+threadIdx.x;    
   int h=blockIdx.y*blockDim.y+threadIdx.y;

     if(w <width && h<height){    //boundary check
        int index= h*width+w; //indexing for the input array in 2D
   
        int rgb_index= index*ch; 
   
       unsigned char red=rgbI[rgb_index];   //getting the pixel for red
       unsigned char green=rgbI[rgb_index+1]; //getting the pixel for green
       unsigned char blue=rgbI[rgb_index+2];  //getting the pixel for blue
   
       gryI[index]= 0.21*red + 0.71*green + 0.07*blue;     
   }
}

__global__ void rgb_to_gry_img_2D_lut(unsigned char* rgbI, int width, int height, int lut_dim, unsigned char* gryI){
 
	int w=blockIdx.x*blockDim.x+threadIdx.x;    
	int h=blockIdx.y*blockDim.y+threadIdx.y;

	if(w <width && h<height){    //boundary check
		int index= h*width+w; //indexing for the input array in 2D
    	
		int rgb_index= index*ch;    
       	unsigned char red=rgbI[rgb_index];   //getting the pixel for red in r
       	unsigned char green=rgbI[rgb_index+1]; //getting the pixel for green
       	unsigned char blue=rgbI[rgb_index+2];  //getting the pixel for blue
   	      
        //gryI[index]= 0.21*red + 0.71*green + 0.07*blue;  	
		gryI[index]= (unsigned char)factors[((red * lut_dim + blue) * lut_dim) + green];
	}
}

__global__ void rgb_to_gry_img_3D(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height, int depth, unsigned char *gry){
  
     int d=blockIdx.x*blockDim.x+threadIdx.x;    
     int w=blockIdx.y*blockDim.y+threadIdx.y;
     int h=blockIdx.z*blockDim.z+threadIdx.z;

     if(w <width && h<height && d<depth){    //boundary check
     	int index= (d*height+h)*width+w;   //indexing in 1D for the 3D input array
   
       	unsigned char red=r[index];           //getting the pixel for red
       	unsigned char green=g[index];      //getting the pixel for green
      	unsigned char blue=b[index];  //getting the pixel for blue
   
      	gry[index]= 0.21*red + 0.71*green + 0.07*blue;     
   }
}



  //3D thread block organization for the image
__global__ void rgb_to_gry_img_3D_lut(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height, int depth, int lut_dim, unsigned char *gry){
  
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

void run_color_conversion_2d(unsigned char *input, unsigned char *out, int width, int height){
	int depth = 3;
    
    unsigned char *d_in;
    unsigned char *d_out; 
   
    float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);   

    //allocationg memory on the GPU for the right image
    cudaMalloc((void **)&d_in, width * height * depth * sizeof(unsigned char));
    cudaMalloc((void **)&d_out, width * height * sizeof(unsigned char));

    //copying image data from Host to device for the right image
    cudaMemcpy(d_in, input, width * height * depth, cudaMemcpyHostToDevice);
    
    //copying image data from Host to device for the left image
    const dim3 blocksize(32,32,1); //declaring the number of threads
    const dim3 gridsize((width-1)/32+1,(height-1)/32+1,1); //declaring the number of blocks
   
    cudaEventRecord(start_time, 0);
    //calling the kernel
    for (int j=0; j<10 ; j++){
    	rgb_to_gry_img_2D<<<gridsize,blocksize>>>(d_in, width, height, d_out);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
    printf("Average execution time 2D: \t\t%f\n",execution_time);

    //coping output data from Device to Host for the right image
    cudaMemcpy(out, d_out, width * height, cudaMemcpyDeviceToHost);

    //freeing the device memory
    cudaFree(d_in);
    cudaFree(d_out);
}

void run_color_conversion_2d_lut(unsigned char *input, unsigned char *out, int width, int height){
	
	int depth = 3;
	int lut_dim = 16;

	float *lut = (float*) malloc(lut_dim * lut_dim * lut_dim * sizeof(float));
    int r,g,b;
    float gray;
    
    for(r=0;r<256;r+=16){
       	for(g=0;g<256;g+=16){
           	for(b=0;b<256;b+=16){
            	gray = 0.21*r + 0.71*g + 0.07*b;				
				lut[(((r / lut_dim) * lut_dim) + (g / lut_dim)) * lut_dim + (b / lut_dim)] = gray;   		
			}
   		}
  	} 

    unsigned char *d_in;
    unsigned char *d_out; 
	
    float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);   

	int size = width * height * depth;
	for(r=0;r<size;r++){	
		input[r] /= lut_dim;			     				
  	}

    //allocationg memory on the GPU for the right image
    cudaMalloc((void **)&d_in, size * sizeof(unsigned char));
    cudaMalloc((void **)&d_out, width * height * sizeof(unsigned char));

    //copying image data from Host to device for the right image
    cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(factors, lut, lut_dim*lut_dim*lut_dim*sizeof(float));
    
    int B=32;
    const dim3 blocksize(B,B,1); //declaring the number of threads
    const dim3 gridsize((width-1)/B+1,(height-1)/B+1,1); //declaring the number of blocks
    
    cudaEventRecord(start_time, 0);
    //calling the kernel
    for (int j=0; j<10 ; j++){
    	rgb_to_gry_img_2D_lut<<<gridsize,blocksize>>>(d_in, width, height, lut_dim, d_out);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
	printf("Average execution time for LUT (2D): \t%f\n",execution_time);
    

    //coping output data from Device to Host for the right image
    cudaMemcpy(out, d_out, width * height, cudaMemcpyDeviceToHost);
    
    //freeing the device memory
    cudaFree(d_in);
    cudaFree(d_out);
}

void run_color_conversion_3d(unsigned char *input, unsigned char *out, int width, int height){

	int depth = 3;
	float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    //unsigned char *h_image = stbi_load(right_image_path, &imgwidth, &imgheight, &imgdepth, 3);
	//unsigned char *h_gray=(unsigned char*)malloc(imgwidth * imgheight * sizeof(unsigned char));
    //split image into separate channels 
    
    unsigned char *h_r=(unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *h_g=(unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *h_b=(unsigned char*)malloc(width * height * sizeof(unsigned char));

	unsigned char *transposed=(unsigned char*)malloc((width * height * depth) * sizeof(unsigned char));	
	transpose(input, transposed, width, height);

	for(int i=0;i<depth;i++){
		for(int j=0; j<height;j++){
		   for(int k=0;k<width;k++) {
               h_r[j * width + k] = transposed[(i * depth + j) * width + k];
               h_g[j * width + k] = transposed[(i * depth + j) * width + k + 1];
               h_b[j * width + k] = transposed[(i * depth + j) * width + k + 2];
      	   }
        }
    }

	unsigned char *d_r;
    unsigned char *d_g;
    unsigned char *d_b;
	unsigned char *d_out;

    cudaMalloc((void **)&d_r,(width * height) * sizeof(unsigned char));
    cudaMalloc((void **)&d_g,(width * height) * sizeof(unsigned char));
    cudaMalloc((void **)&d_b,(width * height) * sizeof(unsigned char));
	cudaMalloc((void **)&d_out,(width * height) * sizeof(unsigned char));
	
	cudaMemcpy(d_r, h_r, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);

	const dim3 blocksize(3,16,16); //declaring the number of threads
    const dim3 gridsize((depth-1)/3+1, (width-1)/16+1,(height-1)/16+1); //declaring the number of blocks

	for (int j=0; j<10 ; j++){
	cudaEventRecord(start_time, 0);
    	rgb_to_gry_img_3D<<<gridsize,blocksize>>>(d_r, d_g, d_b, width, height, depth, d_out);
	}
    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
    printf("Average execution time for 3D: \t\t%f\n",execution_time);
  
    cudaMemcpy(out, d_out, width * height, cudaMemcpyDeviceToHost);

    //freeing the device memory
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_out);
    
    //freeing the host memory
    free(h_r);
    free(h_g);
    free(h_b);
}

void run_color_conversion_3d_lut(unsigned char *input, unsigned char *out, int width, int height){

	int depth = 3;
	int lut_dim = 16;
	float *lut = (float*) malloc(lut_dim * lut_dim * lut_dim * sizeof(float));
    int r,g,b;
    float gray;
    
    for(r=0;r<256;r+=16){
       	for(g=0;g<256;g+=16){
           	for(b=0;b<256;b+=16){
            	gray = 0.21*r + 0.71*g + 0.07*b;			     	
				//printf("%d %d %d %f\n", r, g, b, gray);				
				lut[(((r / lut_dim) * lut_dim) + (g / lut_dim)) * lut_dim + (b / lut_dim)] = gray;   		
			}
   		}
  	}
	

	float execution_time;
    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    unsigned char *h_r=(unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *h_g=(unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *h_b=(unsigned char*)malloc(width * height * sizeof(unsigned char));

	unsigned char *transposed=(unsigned char*)malloc((width * height * depth) * sizeof(unsigned char));	
	transpose(input, transposed, width, height);

	for(int i=0;i<depth;i++){
		for(int j=0; j<height;j++){
		   for(int k=0;k<width;k++) {
               h_r[j * width + k] = transposed[(i * depth + j) * width + k] / lut_dim;
               h_g[j * width + k] = transposed[(i * depth + j) * width + k + 1] / lut_dim;
               h_b[j * width + k] = transposed[(i * depth + j) * width + k + 2] / lut_dim;
      	   }
        }
    }

	unsigned char *d_in;
	unsigned char *d_r;
    unsigned char *d_g;
    unsigned char *d_b;
	unsigned char *d_out;
	
	cudaMalloc((void **)&d_in,(width * height * depth) * sizeof(unsigned char));
    cudaMalloc((void **)&d_r,(width * height) * sizeof(unsigned char));
    cudaMalloc((void **)&d_g,(width * height) * sizeof(unsigned char));
    cudaMalloc((void **)&d_b,(width * height) * sizeof(unsigned char));
	cudaMalloc((void **)&d_out,(width * height) * sizeof(unsigned char));

	cudaMemcpy(d_in, transposed, (width * height * depth) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_r, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(factors, lut, lut_dim*lut_dim*lut_dim*sizeof(float));

	const dim3 blocksize(3,16,16); //declaring the number of threads
    const dim3 gridsize((depth-1)/3+1, (width-1)/16+1,(height-1)/16+1); //declaring the number of blocks

	for (int j=0; j<10 ; j++){
	cudaEventRecord(start_time, 0);
    	rgb_to_gry_img_3D_lut<<<gridsize,blocksize>>>(d_r, d_g, d_b, width, height, depth, lut_dim, d_out);
	}

    cudaDeviceSynchronize();
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&execution_time, start_time, stop_time);
    execution_time /=10.0f;
    printf("Average execution time for LUT (3D): \t%f\n",execution_time);
  
    cudaMemcpy(out, d_out, width * height, cudaMemcpyDeviceToHost);

    //freeing the device memory
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_out);
    
    //freeing the host memory
    free(h_r);
    free(h_g);
    free(h_b);
}

void run_color_conversion(unsigned char *input, unsigned char *out, int width, int height, int type){
	if(type == 0){ 			// 2D version
		printf("Using 2D color conversion\n");		
		run_color_conversion_2d(input, out, width, height);
	}else if(type == 1){	// 2D + LUT	
		run_color_conversion_2d_lut(input, out, width, height);
		printf("Using 2D + LUT (constant memory )color conversion\n");
	}else if(type == 2){	// 3D
		printf("Using 3D color conversion\n");
		run_color_conversion_3d(input, out, width, height);
	}else if(type == 3){	// 3D + LUT
		printf("Using 3D + LUT (constant memory )color conversion\n");
		run_color_conversion_3d_lut(input, out, width, height);
	}else{
		printf("No version like this!!!\n");
		exit(0);
	}
}



