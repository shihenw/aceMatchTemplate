#ifndef MYOWNMATCHTEMPLATE_H
#define MYOWNMATCHTEMPLATE_H

#include <cuda.h>
#include <cufft.h>
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

// inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c) {
//     fComplex t = {c *(a.x * b.x - a.y * b.y), c *(a.y * b.x + a.x * b.y)};
//     a = t;
// }

// __global__ void modulateAndNormalize_kernel(fComplex *d_Dst, fComplex *d_Src, int dataSize, float c) {
//     const int i = blockDim.x * blockIdx.x + threadIdx.x;

//     if (i >= dataSize) {
//         return;
//     }

//     fComplex a = d_Src[i];
//     fComplex b = d_Dst[i];

//     mulAndScale(a, b, c);

//     d_Dst[i] = a;
// }

// void modulateAndNormalize(fComplex *d_Dst, fComplex *d_Src, int fftH, int fftW, int padding) {
//     assert(fftW % 2 == 0);
//     const int dataSize = fftH * (fftW / 2 + padding);

//     modulateAndNormalize_kernel<<<updiv(dataSize, 256), 256>>>(d_Dst, d_Src, dataSize, 1.0f / (float)(fftW *fftH));
//     //getLastCudaError("modulateAndNormalize() execution failed\n");
// }

// void check(int r){
// 	if(r != 0){
// 		printf("check failed!\n");
// 	}
// }

// void myOwnMatchTemplate(int batch_size, int num_template, int image_width, int image_height,
// 	                    float* d_batch_images, float* d_all_templates, 
// 	                    int* d_templ_height, int* d_templ_width, 
//                         float* d_convoluted, float* d_integral_img, float* d_templ_sqsum, float* d_scores){

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
// 	cufftHandle fftPlanFwd, fftPlanInv;
// 	float* h_Kernel, *h_ResultGPU;
// 	float* d_Kernel;
// 	fComplex* d_DataSpectrum, *d_KernelSpectrum;

// 	int kernelH = 4, kernelW = 2;
// 	const int fftH = image_height + kernelH - 1;
//     const int fftW = image_width + kernelW - 1;
	
// 	h_Kernel    = (float *)malloc(kernelH * kernelW * sizeof(float));
//     h_ResultGPU = (float *)malloc(fftH    * fftW * sizeof(fComplex));
// 	cudaMalloc(&d_Kernel, kernelH * kernelW * sizeof(float));
// 	cudaMalloc(&d_DataSpectrum,   fftH * (fftW / 2 + 1) * sizeof(fComplex));
//     cudaMalloc(&d_KernelSpectrum, 4 * (2 / 2 + 1) * sizeof(fComplex));

// 	printf("...creating R2C & C2R FFT plans for %i x %i\n", fftH, fftW);
	
//     check(cufftPlan2d(&fftPlanFwd, kernelH, kernelW, CUFFT_R2C));
//     check(cufftPlan2d(&fftPlanInv, kernelH, kernelW, CUFFT_C2R));

//     float temp[9] = {3, 5, 2, 1, 8, 9, 2, 3, 0};
//     for (int i = 0; i < kernelH * kernelW; i++) {
//         h_Kernel[i] = temp[i];
//     }
//     cudaError_t R = cudaMemcpy(d_Kernel, h_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyHostToDevice);
//     if (R != cudaSuccess) {
//     	printf("cudaMemcpy not success!\n");
//     }

//     //check(cufftExecR2C(fftPlanFwd, (cufftReal *)d_batch_images, (cufftComplex *)d_DataSpectrum));
//     check(cufftExecR2C(fftPlanFwd, (cufftReal *)d_Kernel, (cufftComplex *)d_KernelSpectrum));

//     cudaMemcpy(h_ResultGPU, d_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyDeviceToHost);
//     for(int i = 0; i < kernelH * kernelW; i++){
//     	printf("%f ", h_ResultGPU[i]);
//     }
//     printf("\n");
//     cudaMemcpy(h_ResultGPU, d_KernelSpectrum, kernelH * (kernelW/2+1) * sizeof(fComplex), cudaMemcpyDeviceToHost);
//     for(int i = 0; i < kernelH * (kernelW/2+1); i++){
//     	printf("%f ", h_ResultGPU[i]);
//     }
//     printf("\n");
//     cufftExecC2R(fftPlanInv, (cufftComplex *)d_KernelSpectrum, (cufftReal *)d_Kernel);
//     cudaMemcpy(h_ResultGPU, d_Kernel, kernelH * kernelW * sizeof(float), cudaMemcpyDeviceToHost);
//     for(int i = 0; i < kernelH * kernelW; i++){
//     	printf("%f ", h_ResultGPU[i]);
//     }
//     printf("\n");

//     //modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);
//     //check(cufftExecC2R(fftPlanInv, (cufftComplex *)d_DataSpectrum, (cufftReal *)d_convoluted));

//     //cudaMemcpy(h_ResultGPU, d_convoluted, fftH * fftW * sizeof(float), cudaMemcpyDeviceToHost);

//     //print something
//     // for(int j=0; j < 10; j++){
//     //     for(int i=0; i < 40; i++){
//     //         printf("%d ", int(h_ResultGPU[j*image_width+i]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("\n");

// 	//4. SSD_NORMED

// 	//5. maximum location
// 	cudaDeviceSynchronize();
// }

#endif