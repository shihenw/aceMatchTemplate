#ifndef MYOWNMATCHTEMPLATE_H
#define MYOWNMATCHTEMPLATE_H

#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef double2 dComplex;

__global__ void sqrIntegralKernelPass1(int batch_size, int image_width, int image_height, 
	                                   int padded_width, int padded_height, double* d_batch_images){
	//square, and scan horizontally
	//global thread idx is global # of row
	//d_batch_images: 	[w * h * batch_size]
	//d_integral_img:	[w * h * batch_size]
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(thread_index < image_height * batch_size){
		int img_index = thread_index / image_height;
		int row_index = thread_index % image_height;
		//if(row_index == 0) printf("row_index = %d, image_width = %d\n", row_index, image_width);
		double sum = 0;
		double temp = 0;
		int pitch2 = padded_height * padded_width;
		for(int i = 0; i < image_width + 1; i++){ //note +1 here
			int index = img_index * pitch2 + row_index * padded_width + i;
			temp = d_batch_images[index];
			d_batch_images[index] = sum;
			sum += temp * temp;
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
	                                   int padded_width, int padded_height, double* d_batch_images){
	//scan vertically
	//global thread idx is global # of column
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int img_index = thread_index / (image_width + 1);
	int col_index = thread_index % (image_width + 1);
	if(thread_index < (image_width + 1) * batch_size){
	//if(col_index == 0){
		//printf("col_index: %d, img_index: %d, width_index: %d\n", col_index, img_index, width_index);
		//if(row_index == 0) printf("row_index = %d, image_width = %d\n", row_index, image_width);
		double sum = 0;
		double temp = 0;
		int pitch2 = padded_height * padded_width;
		for(int i = 0; i < image_height + 1; i++) { //i is row index
			int index = img_index * pitch2 + i * padded_width + col_index;
			temp = d_batch_images[index];
			d_batch_images[index] = sum;
			sum += temp;
			// if(row_index == 0){
			// 	printf("%.3f ", sum);
			// }
		}
	}
}

inline __device__ void mulAndScale(dComplex &a, const dComplex &b, const double &c) {
    dComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    a = t;
}

__global__ void modulateAndNormalize_kernel(dComplex *d_Dst, dComplex *d_templ_spec, dComplex *d_im_spec, 
	                                        int total_rows, int pitch1, int padded_height, double c) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread id (0 - 32 * total_threads)
    int duty_row = i >> 5; //i / 32;
    int duty_row_onImage = duty_row % padded_height;
    if (duty_row >= total_rows) {
        return;
    }
    int offset = i % 32;
    //double c = double(1) / double(padded_width * padded_height);
    
    //                          |---total_rows---|
    //note: d_templ_spec is in [96 * padded_height * (padded_width/2+1)]
    //      d_im_spec is in    [     padded_height * (padded_width/2+1)]
    for(int w = offset; w < pitch1; w += 32){ //pitch1 is in dComplex unit (2 * 4B)
    	int index_templ = duty_row * pitch1 + w;
    	int index_im = duty_row_onImage * pitch1 + w;
	    dComplex a = d_templ_spec[index_templ];
	    dComplex b = d_im_spec[index_im];

	 //    if(duty_row == 14 * padded_height && w == 0){
		//     printf("a=%f\nb=%f\na*b=%f\na*b/c=%f\n", a.x, b.x, a.x*b.x, a.x*b.x/padded_width/padded_height);
		// }
		// if(i == 0 && w > pitch1 - 3){
		// 	printf("a = %.3f + %.3fi, b = %.3f + %.3fi\n", a.x, a.y, b.x, b.y);
		// }
	    
	    mulAndScale(a, b, c);
	    d_Dst[index_templ] = a;
	}
}

__device__ double normAcc_SQDIFF(double num, double denum){
    if (::fabs(num) < denum)
        return num / denum;
    if (::fabs(num) < denum * 1.125f)
        return num > 0 ? 1 : -1;
    return 1;
}


__global__ void getScoreMap_kernel(double* d_scoreMap, double* d_batch_images, double* d_convolved, double* d_templ_sqsum, 
	                               int* d_templ_width, int* d_templ_height, 
                                   int num_template, int padded_width, int padded_height, 
                                   int image_width, int image_height, int image_index){
	const int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread id (0 - 32 * total_threads)
    int duty_row = i >> 5; //i / 32;
    int duty_template = duty_row / padded_height;
    int h = duty_row % padded_height; //duty_row_local

    int kh = d_templ_height[duty_template];
    int kw = d_templ_width[duty_template];

    if (duty_row >= num_template * padded_height || h >= image_height - kh + 1) {
        return;
    }
    int offset = i % 32;

    int pitch1 = padded_width;
    int pitch2 = padded_height * pitch1;

    double* d_integral_img = d_batch_images + image_index * pitch2;
    double templ_sqsum = d_templ_sqsum[duty_template];
    double templ_sqsum_root = sqrt(templ_sqsum);

    // if(duty_row == 14*padded_height+151 && offset == 0){
    // 	printf("templ_sqsum = %f, templ_sqsum_root = %f\n", templ_sqsum, templ_sqsum_root);
    // }

    for(int w = offset; w < image_width - kw + 1; w += 32){
    	//d_scoreMap:  num_template * padded_height * padded_width
    	//d_convolved: num_template * padded_height * padded_width
    	//d_batch_images: batch_image_size * padded_height * padded_width
    	
    	//index for score map
    	int index1 = duty_row * pitch1 + w;
    	
    	//index for integral image
    	int index2_ll = h * pitch1 + w;
    	int index2_rr = (h+kh) * pitch1 + (w+kw);
    	int index2_lr = h * pitch1 + (w+kw);
    	int index2_rl = (h+kh) * pitch1 + w;

    	//index for convolution map
    	int index3 = duty_template * pitch2 + (h + kh - 1) * pitch1 + (w + kw - 1);

	    double image_sqsum = d_integral_img[index2_rr] - d_integral_img[index2_lr] - d_integral_img[index2_rl] + d_integral_img[index2_ll];
	    double image_sqsum_root = sqrt(image_sqsum);
	    double conv = d_convolved[index3];

	    d_scoreMap[index1] = normAcc_SQDIFF(image_sqsum - 2.f * conv + templ_sqsum, image_sqsum_root * templ_sqsum_root);

	    // if(duty_row == 14*padded_height+151 && w == 207){
	    // 	printf("w = %d, image_sqsum = %f, conv = %f, templ_sqsum = %f, score = %f\n", 
	    // 		         w, image_sqsum,      conv,      templ_sqsum, d_scoreMap[index1]);
	    // 	printf("corners: %f, %f, %f, %f\n", d_integral_img[index2_ll], d_integral_img[index2_rl], 
	    // 		                                d_integral_img[index2_lr], d_integral_img[index2_rr]);
	    // }
	}
}

__global__ void rowReduceMin_kernel(double* d_scoreMap, int* d_templ_width, int* d_templ_height, 
	                                int num_template, int padded_width, int padded_height, 
	                                int image_width, int image_height){
	
	const int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread id (0 - 32 * total_threads)
	int duty_row = i >> 5; //i / 32; //duty row indexing is skipping [301 - 512] row
    int duty_template = duty_row / image_height;
    int h = duty_row % image_height; //duty_row_local

    int width_of_scoreMap = image_width - d_templ_width[duty_template] + 1;
    int height_of_scoreMap = image_height - d_templ_height[duty_template] + 1;

    if (duty_row < num_template * image_width && h < height_of_scoreMap) {
        int offset = i % 32;
	    int warp_id = (i % 256) / 32; //0-7
	    int pitch1 = padded_width;
	    int pitch2 = padded_height * pitch1;

	    __shared__ double row_minVals[8][32];//256
	    __shared__ int row_xargmin[8][32];

	    double minVal = 1.1; //maximum is 1
	    int xargmin = 0;
	    int index = duty_template * pitch2 + h * pitch1;
	    for(int w = offset; w < width_of_scoreMap; w += 32){
	    	double read = d_scoreMap[index+w];
	    	//if(duty_row == 3*image_height && offset == 0){
	    	//	printf("d_scoreMap[%d] = %f\n", index, read);
	    	//}
	    	if(read < minVal){
	    		minVal = read;
	    		xargmin = w;
	    	}
	    }
	    row_minVals[warp_id][offset] = minVal;
	    row_xargmin[warp_id][offset] = xargmin;
	    //__syncthreads(); //ensure share memory are all filled

	    // if(offset == 0 && duty_row == 0){
	    // 	for(int i = 0; i < 32; i++) printf("%f ", row_minVals[warp_id][i]);
	    // 	printf("\n");
	    // 	for(int i = 0; i < 32; i++) printf("%d ", row_xargmin[warp_id][i]);
	    // 	printf("\n");
	    // }
	    // if(duty_row == 3*image_height){
	    // 	printf("thread %d: minval %f, index %d, warp_id %d\n", offset, minVal, xargmin, warp_id);
	    // }

	    if(offset % 2 == 0){
	    	if(row_minVals[warp_id][offset] > row_minVals[warp_id][offset+1]){
	    		row_minVals[warp_id][offset] = row_minVals[warp_id][offset+1];
	    		row_xargmin[warp_id][offset] = row_xargmin[warp_id][offset+1];
	    	}
	    }
	    if(offset % 4 == 0){
	    	if(row_minVals[warp_id][offset] > row_minVals[warp_id][offset+2]){
	    		row_minVals[warp_id][offset] = row_minVals[warp_id][offset+2];
	    		row_xargmin[warp_id][offset] = row_xargmin[warp_id][offset+2];
	    	}
	    }
	    if(offset % 8 == 0){
	    	if(row_minVals[warp_id][offset] > row_minVals[warp_id][offset+4]){
	    		row_minVals[warp_id][offset] = row_minVals[warp_id][offset+4];
	    		row_xargmin[warp_id][offset] = row_xargmin[warp_id][offset+4];
	    	}
	    }
	    if(offset % 16 == 0){
	    	if(row_minVals[warp_id][offset] > row_minVals[warp_id][offset+8]){
	    		row_minVals[warp_id][offset] = row_minVals[warp_id][offset+8];
	    		row_xargmin[warp_id][offset] = row_xargmin[warp_id][offset+8];
	    	}
	    }
	    if(offset % 32 == 0){
	    	if(row_minVals[warp_id][offset] > row_minVals[warp_id][offset+16]){
	    		row_minVals[warp_id][offset] = row_minVals[warp_id][offset+16];
	    		row_xargmin[warp_id][offset] = row_xargmin[warp_id][offset+16];
	    	}
	    	d_scoreMap[index] = row_xargmin[warp_id][0];
	    	d_scoreMap[index + 1] = (double)row_minVals[warp_id][0];
	    }
    }
}


__global__ void columnReduceMin_kernel(double* d_scoreMap, int* d_templ_width, int* d_templ_height, 
									double* d_minval, int* d_argmin_x, int* d_argmin_y,
	                                int num_template, int padded_width, int padded_height, 
	                                int image_width, int image_height){
	//256 threads per template
	const int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread id
	int duty_template = i >> 5; //i / 32;

	if(duty_template < num_template){
	    int offset = i % 32;
	    int warp_id = (i % 256) / 32; //0-7
	    int pitch1 = padded_width;
	    int pitch2 = padded_height * pitch1;

	    int height_of_scoreMap = image_height - d_templ_height[duty_template] + 1;

	    __shared__ double col_minVals[8][32];//256
	    __shared__ int col_xargmin[8][32];
	    __shared__ int col_yargmin[8][32];

	    double minVal = 1.1; //maximum is 1
	    int xargmin = 0;
	    int yargmin = 0;

	    for(int h = offset; h < height_of_scoreMap; h += 32){
	    	int index = duty_template * pitch2 + h * pitch1;
	    	double read = d_scoreMap[index+1];
	    	if(read < minVal){
	    		minVal = read;
	    		xargmin = (int)(d_scoreMap[index] + 0.5);
	    		yargmin = h;
	    	}
	    }

	    // if(duty_template == 0){
	    // 	printf("thread %d: minval %f, xargmin %d, yargmin %d, warp_id %d\n", 
	    // 		offset, minVal, xargmin, yargmin, warp_id);
	    // }
	    col_minVals[warp_id][offset] = minVal;
	    col_xargmin[warp_id][offset] = xargmin;
	    col_yargmin[warp_id][offset] = yargmin;

	    if(offset % 2 == 0){
	    	if(col_minVals[warp_id][offset] > col_minVals[warp_id][offset+1]){
	    		col_minVals[warp_id][offset] = col_minVals[warp_id][offset+1];
	    		col_xargmin[warp_id][offset] = col_xargmin[warp_id][offset+1];
	    		col_yargmin[warp_id][offset] = col_yargmin[warp_id][offset+1];
	    	}
	    }
	    if(offset % 4 == 0){
	    	if(col_minVals[warp_id][offset] > col_minVals[warp_id][offset+2]){
	    		col_minVals[warp_id][offset] = col_minVals[warp_id][offset+2];
	    		col_xargmin[warp_id][offset] = col_xargmin[warp_id][offset+2];
	    		col_yargmin[warp_id][offset] = col_yargmin[warp_id][offset+2];
	    	}
	    }
	    if(offset % 8 == 0){
	    	if(col_minVals[warp_id][offset] > col_minVals[warp_id][offset+4]){
	    		col_minVals[warp_id][offset] = col_minVals[warp_id][offset+4];
	    		col_xargmin[warp_id][offset] = col_xargmin[warp_id][offset+4];
	    		col_yargmin[warp_id][offset] = col_yargmin[warp_id][offset+4];
	    	}
	    }
	    if(offset % 16 == 0){
	    	if(col_minVals[warp_id][offset] > col_minVals[warp_id][offset+8]){
	    		col_minVals[warp_id][offset] = col_minVals[warp_id][offset+8];
	    		col_xargmin[warp_id][offset] = col_xargmin[warp_id][offset+8];
	    		col_yargmin[warp_id][offset] = col_yargmin[warp_id][offset+8];
	    	}
	    }
	    if(offset % 32 == 0){
	    	if(col_minVals[warp_id][offset] > col_minVals[warp_id][offset+16]){
	    		col_minVals[warp_id][offset] = col_minVals[warp_id][offset+16];
	    		col_xargmin[warp_id][offset] = col_xargmin[warp_id][offset+16];
	    		col_yargmin[warp_id][offset] = col_yargmin[warp_id][offset+16];
	    	}
	    	d_minval[duty_template] = col_minVals[warp_id][0];
	    	d_argmin_x[duty_template] = col_xargmin[warp_id][0];
	    	d_argmin_y[duty_template] = col_yargmin[warp_id][0];
	    }
	}
}

void check(int r){
	if(r != 0){
		printf("check failed!\n");
	}
}

void test(){

	cufftHandle fftPlanFwd, fftPlanInv;
	int kernelH = 4;
	int kernelW = 8;
	double* h_temp, *h_temp_spec, *h_image, *h_image_spec, *h_spec_mul, *h_conv;
	double* d_temp, *d_temp_spec, *d_image, *d_image_spec, *d_spec_mul, *d_conv;

// 	dComplex* d_DataSpectrum, *d_KernelSpectrum;

// 	int kernelH = 4, kernelW = 2;
// 	const int fftH = image_height + kernelH - 1;
//     const int fftW = image_width + kernelW - 1;
	
	h_temp       = (double *)malloc(kernelH * kernelW * sizeof(double));
    h_temp_spec  = (double *)malloc(kernelH * (kernelW/2+1) * sizeof(dComplex));
    h_image      = (double *)malloc(kernelH * kernelW * sizeof(double));
    h_image_spec = (double *)malloc(kernelH * (kernelW/2+1) * sizeof(dComplex));
    h_spec_mul   = (double *)malloc(kernelH * (kernelW/2+1) * sizeof(dComplex));
    h_conv       = (double *)malloc(kernelH * kernelW * sizeof(double));

 	cudaMalloc(&d_temp, 		kernelH * kernelW * sizeof(double));
 	cudaMalloc(&d_temp_spec, 	kernelH * (kernelW/2+1) * sizeof(dComplex));
 	cudaMalloc(&d_image, 		kernelH * kernelW * sizeof(double));
 	cudaMalloc(&d_image_spec, 	kernelH * (kernelW/2+1) * sizeof(dComplex));
 	cudaMalloc(&d_spec_mul, 	kernelH * (kernelW/2+1) * sizeof(dComplex));
 	cudaMalloc(&d_conv, 		kernelH * kernelW * sizeof(double));
	
    check(cufftPlan2d(&fftPlanFwd, kernelH, kernelW, CUFFT_R2C));
    check(cufftPlan2d(&fftPlanInv, kernelH, kernelW, CUFFT_C2R));

    double temp[32] = {3, 5, 9, 2, 2, 0, 0, 0,
    	              8, 9, 1, 4, 9, 0, 0, 0, 
    	              0, 0, 0, 0, 0, 0, 0, 0,
    	              0, 0, 0, 0, 0, 0, 0, 0};

    double image[32] = {100, 50, 12, 24, 0, 0, 0, 0, 
    	                40, 90, 98, 63, 0, 0, 0, 0,
    	                20, 15, 17, 29, 0, 0, 0, 0,
    	            	 0,  0,  0,  0, 0, 0, 0, 0};

    for (int i = 0; i < kernelH * kernelW; i++) {
        h_temp[i] = temp[i];
        h_image[i] = image[i];
    }
    cudaMemcpy(d_temp, h_temp, kernelH * kernelW * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, h_image, kernelH * kernelW * sizeof(double), cudaMemcpyHostToDevice);

    check(cufftExecD2Z(fftPlanFwd, (cufftDoubleReal *)d_temp, (cufftDoubleComplex *)d_temp_spec));
    check(cufftExecD2Z(fftPlanFwd, (cufftDoubleReal *)d_image, (cufftDoubleComplex *)d_image_spec));

    cudaMemcpy(h_temp_spec, d_temp_spec, kernelH * (kernelW/2 + 1) * sizeof(dComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image_spec, d_image_spec, kernelH * (kernelW/2 + 1) * sizeof(dComplex), cudaMemcpyDeviceToHost);
    
    printf("temp spec:\n");
    for(int i = 0; i < kernelH * (kernelW/2+1) * 2; i++){
    	printf("%f ", h_temp_spec[i]);
    }
    printf("\nimage spec:\n");
    for(int i = 0; i < kernelH * (kernelW/2+1) * 2; i++){
    	printf("%f ", h_image_spec[i]);
    }
    printf("\n");
    
    modulateAndNormalize_kernel<<<1,kernelH>>>((dComplex*) d_spec_mul, (dComplex*) d_image_spec, (dComplex*) d_temp_spec, 
    	                                       kernelH, (kernelW/2)+1, kernelH, 1.0f/double(kernelW*kernelH));
 
    cudaMemcpy(h_spec_mul, d_spec_mul, kernelH * (kernelW/2+1) * sizeof(dComplex), cudaMemcpyDeviceToHost);
    printf("spec product\n");
    for(int i = 0; i < kernelH * (kernelW/2+1) * 2; i++){
    	printf("%f ", h_spec_mul[i]);
    }
    printf("\n");

	check(cufftExecZ2D(fftPlanInv, (cufftDoubleComplex *)d_spec_mul, (cufftDoubleReal *)d_conv));
	cudaMemcpy(h_conv, d_conv, kernelW * kernelH * sizeof(double), cudaMemcpyDeviceToHost);
    printf("conv result:\n");
    for(int i = 0; i < kernelH * kernelW; i++){
        printf("%f ", h_conv[i]);
    }
    printf("\n");
}

#endif