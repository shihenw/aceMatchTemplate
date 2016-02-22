#ifndef MYOWNMATCHTEMPLATE_H
#define MYOWNMATCHTEMPLATE_H

#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef float2 fComplex;

__global__ void sqrIntegralKernelPass1(int batch_size, int image_width, int image_height, 
	                                   int padded_width, int padded_height, float* d_batch_images){
	//square, and scan horizontally
	//global thread idx is global # of row
	//d_batch_images: 	[w * h * batch_size]
	//d_integral_img:	[w * h * batch_size]
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_index < image_height * batch_size){
		int img_index = thread_index / image_height;
		int row_index = thread_index % image_height;
		//if(row_index == 0) printf("row_index = %d, image_width = %d\n", row_index, image_width);
		float sum = 0;
		float temp = 0;
		int pitch2 = padded_height * padded_width;
		for(int i = 0; i < image_width; i++){
			int index = img_index * pitch2 + row_index * padded_width + i;
			temp = d_batch_images[index];
			sum += temp * temp;
			d_batch_images[index] = sum;
			// if(row_index == 0){
			// 	printf("%.3f ", sum);
			// }
		}
		// if(row_index == 0){
		// 	printf("\n");
		// }
	}
}

__global__ void sqrIntegralKernelPass2(int batch_size, int image_width, int image_height, 
	                                   int padded_width, int padded_height, float* d_batch_images){
	//scan vertically
	//global thread idx is global # of column
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int img_index = thread_index / image_width;
	int col_index = thread_index % image_width;
	if(thread_index < image_width * batch_size){
	//if(col_index == 0){
		//printf("col_index: %d, img_index: %d, width_index: %d\n", col_index, img_index, width_index);
		//if(row_index == 0) printf("row_index = %d, image_width = %d\n", row_index, image_width);
		float sum = 0;
		int pitch2 = padded_height * padded_width;
		for(int i = 0; i < image_height; i++){ //i is row index
			int index = img_index * pitch2 + i * padded_width + col_index;
			sum += d_batch_images[index];
			d_batch_images[index] = sum;
			// if(row_index == 0){
			// 	printf("%.3f ", sum);
			// }
		}
	}
}

inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c) {
    fComplex t = {c *(a.x * b.x - a.y * b.y), c *(a.y * b.x + a.x * b.y)};
    a = t;
}

__global__ void modulateAndNormalize_kernel(fComplex *d_Dst, fComplex *d_Src1, fComplex *d_Src2, int total_rows, int pitch1, float c) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= total_rows) {
        return;
    }

    for(int w = 0; w < pitch1; w++){ //pitch1 is in fComplex unit (2 * 4B)
    	int index = i * pitch1 + w;
	    fComplex a = d_Src1[index];
	    fComplex b = d_Src2[index];

		// if(i == 0 && w > pitch1 - 3){
		// 	printf("a = %.3f + %.3fi, b = %.3f + %.3fi\n", a.x, a.y, b.x, b.y);
		// }

	    mulAndScale(a, b, c);
	    d_Dst[index] = a;
	}
}

void check(int r){
	if(r != 0){
		printf("check failed!\n");
	}
}

void test(){

// 	//d_scores:		 	[w * h * num_templ * batch_size]
//     //d_templ_sqsum: 	[num_templ] 
//     //d_integral_img:	[w * h * batch_size]
//     //d_convoluted:	 	[w * h * num_templ * batch_size]
//     //d_batch_images: 	[w * h * batch_size]
//     //d_all_templates:	[w' * h' * num_templ]
//     //d_templ_height:	[num_templ]
//     //d_templ_width:	[num_templ]
//     //batch_size: 128

// 	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*1024*1024);

// 	//1. sqrIntegral
// 	int total_rows = image_height * batch_size; //300 * 128
// 	int total_cols = image_width * batch_size;  //426 * 128
// 	int numBlocks = updiv(total_rows, numThreadsPerBlock);
// 	printf("Launching sqrIntegralKernelPass1<<<%d, %d>>>\n", numBlocks, numThreadsPerBlock);
// 	sqrIntegralKernelPass1<<<numBlocks, numThreadsPerBlock>>>(batch_size, image_width, image_height, d_batch_images, d_integral_img);
	
// 	// int size = image_height * image_width * batch_size;
// 	// float* h_integral_img_debug = new float[size];
//  //    cudaMemcpy(h_integral_img_debug, d_integral_img, size * sizeof(float), cudaMemcpyDeviceToHost);

//  //    for(int j=0; j < 10; j++){
//  //        for(int i=0; i < 40; i++){
//  //            printf("%d ", int(h_integral_img_debug[j*image_width+i])%10);
//  //        }
//  //        printf("\n");
//  //    }
//  //    printf("\n");

// 	numBlocks = updiv(total_cols, numThreadsPerBlock);
// 	printf("Launching sqrIntegralKernelPass2<<<%d, %d>>>\n", numBlocks, numThreadsPerBlock);
// 	sqrIntegralKernelPass2<<<numBlocks, numThreadsPerBlock>>>(batch_size, image_width, image_height, d_batch_images, d_integral_img);

// 	// cudaMemcpy(h_integral_img_debug, d_integral_img, size * sizeof(float), cudaMemcpyDeviceToHost);
// 	// for(int j=0; j < 10; j++){
//  //        for(int i=0; i < 40; i++){
//  //            printf("%d ", int(h_integral_img_debug[j*image_width+i])%10);
//  //        }
//  //        printf("\n");
//  //    }
//  //    printf("\n");
	
// 	//2. template_sqsum 
// 	//(done outside)

// 	//3. convolution
	cufftHandle fftPlanFwd, fftPlanInv;
	int kernelH = 4;
	int kernelW = 8;
	float* h_temp, *h_temp_spec, *h_image, *h_image_spec, *h_spec_mul, *h_conv;
	float* d_temp, *d_temp_spec, *d_image, *d_image_spec, *d_spec_mul, *d_conv;

// 	fComplex* d_DataSpectrum, *d_KernelSpectrum;

// 	int kernelH = 4, kernelW = 2;
// 	const int fftH = image_height + kernelH - 1;
//     const int fftW = image_width + kernelW - 1;
	
	h_temp       = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_temp_spec  = (float *)malloc(kernelH * (kernelW/2+1) * sizeof(fComplex));
    h_image      = (float *)malloc(kernelH * kernelW * sizeof(float));
    h_image_spec = (float *)malloc(kernelH * (kernelW/2+1) * sizeof(fComplex));
    h_spec_mul   = (float *)malloc(kernelH * (kernelW/2+1) * sizeof(fComplex));
    h_conv       = (float *)malloc(kernelH * kernelW * sizeof(float));

 	cudaMalloc(&d_temp, 		kernelH * kernelW * sizeof(float));
 	cudaMalloc(&d_temp_spec, 	kernelH * (kernelW/2+1) * sizeof(fComplex));
 	cudaMalloc(&d_image, 		kernelH * kernelW * sizeof(float));
 	cudaMalloc(&d_image_spec, 	kernelH * (kernelW/2+1) * sizeof(fComplex));
 	cudaMalloc(&d_spec_mul, 	kernelH * (kernelW/2+1) * sizeof(fComplex));
 	cudaMalloc(&d_conv, 		kernelH * kernelW * sizeof(float));
	
    check(cufftPlan2d(&fftPlanFwd, kernelH, kernelW, CUFFT_R2C));
    check(cufftPlan2d(&fftPlanInv, kernelH, kernelW, CUFFT_C2R));

    float temp[32] = {3, 5, 9, 2, 2, 0, 0, 0,
    	              8, 9, 1, 4, 9, 0, 0, 0, 
    	              0, 0, 0, 0, 0, 0, 0, 0,
    	              0, 0, 0, 0, 0, 0, 0, 0};

    float image[32] = {100, 50, 12, 24, 0, 0, 0, 0, 
    	                40, 90, 98, 63, 0, 0, 0, 0,
    	                20, 15, 17, 29, 0, 0, 0, 0,
    	            	 0,  0,  0,  0, 0, 0, 0, 0};

    for (int i = 0; i < kernelH * kernelW; i++) {
        h_temp[i] = temp[i];
        h_image[i] = image[i];
    }
    cudaMemcpy(d_temp, h_temp, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, h_image, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice);

    check(cufftExecR2C(fftPlanFwd, (cufftReal *)d_temp, (cufftComplex *)d_temp_spec));
    check(cufftExecR2C(fftPlanFwd, (cufftReal *)d_image, (cufftComplex *)d_image_spec));

    cudaMemcpy(h_temp_spec, d_temp_spec, kernelH * (kernelW/2 + 1) * sizeof(fComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image_spec, d_image_spec, kernelH * (kernelW/2 + 1) * sizeof(fComplex), cudaMemcpyDeviceToHost);
    
    printf("temp spec:\n");
    for(int i = 0; i < kernelH * (kernelW/2+1) * 2; i++){
    	printf("%f ", h_temp_spec[i]);
    }
    printf("\nimage spec:\n");
    for(int i = 0; i < kernelH * (kernelW/2+1) * 2; i++){
    	printf("%f ", h_image_spec[i]);
    }
    printf("\n");
    
    modulateAndNormalize_kernel<<<1,kernelH>>>((fComplex*) d_spec_mul, (fComplex*) d_image_spec, (fComplex*) d_temp_spec, 
    	                                       kernelH, (kernelW/2)+1, 1.0/float(kernelW*kernelH));
 
    cudaMemcpy(h_spec_mul, d_spec_mul, kernelH * (kernelW/2+1) * sizeof(fComplex), cudaMemcpyDeviceToHost);
    printf("spec product\n");
    for(int i = 0; i < kernelH * (kernelW/2+1) * 2; i++){
    	printf("%f ", h_spec_mul[i]);
    }
    printf("\n");

	check(cufftExecC2R(fftPlanInv, (cufftComplex *)d_spec_mul, (cufftReal *)d_conv));
	cudaMemcpy(h_conv, d_conv, kernelW * kernelH * sizeof(float), cudaMemcpyDeviceToHost);
    printf("conv result:\n");
    for(int i = 0; i < kernelH * kernelW; i++){
        printf("%f ", h_conv[i]);
    }
    printf("\n");

// 	//4. SSD_NORMED

// 	//5. maximum location
// 	cudaDeviceSynchronize();
}

#endif