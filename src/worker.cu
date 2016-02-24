#include <fstream>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "worker.h"
#include "util/CycleTimer.h"
#include "myOwnMatchTemplate.cuh"

#define numThreadsPerBlock 256

typedef double2 dComplex;

int updiv(int a, int b){
    return (a+b-1)/b;
}

Worker::Worker(char *argv[]) {
    //set device id first
    startTime = CycleTimer::currentSeconds();
    cudaSetDevice(0); //speed reply on system memory?
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("Time for setID: %f sec.\n", endTime-startTime);
    printf("run: using GPU %d \n", 2);
    fflush(stdout);

    // work spec: read from arguments
    template_list = argv[4];
    template_folder = argv[5];
    image_name = argv[6];
    result_filename = argv[7];
    begin_image_num = atoi(argv[8]);
    end_image_num = atoi(argv[9]);
    device_id = atoi(argv[1]);
    num_devices = atoi(argv[3]);
    device_order = atoi(argv[2]);

    // output
    score.resize(num_template);
    position.resize(num_template);
    template_name.resize(num_template);

    //allocate all CUDA memory
    double GPU_MEM = 0;
    // i) for template
    h_templ_width = new int [num_template];
    h_templ_height = new int [num_template];
    cudaMalloc(&d_templ_width, num_template * sizeof(int));         GPU_MEM += num_template * sizeof(int);
    cudaMalloc(&d_templ_height, num_template * sizeof(int));        GPU_MEM += num_template * sizeof(int);
    h_templ_sqsum = new double [num_template];
    cudaMalloc(&d_templ_sqsum, num_template * sizeof(double));       GPU_MEM += num_template * sizeof(double);
    h_templates = new double [padded_height * padded_width];
    // temp space
    cudaMalloc(&d_templates, padded_height * padded_width * sizeof(double));    GPU_MEM += padded_height * padded_width * sizeof(double);
    h_all_templates_spectrum = new double [2 * num_template * padded_height * (padded_width/2 + 1)];
    cudaMalloc(&d_all_templates_spectrum, num_template * padded_height * (padded_width/2 + 1) * sizeof(dComplex));
                                                                                GPU_MEM += num_template * padded_height * (padded_width/2 + 1) * sizeof(dComplex);
    // ii) for image
    cudaMalloc(&d_batch_images, padded_height * padded_width * batch_image_size * sizeof(double));       GPU_MEM += padded_height * padded_width * batch_image_size * sizeof(double);
    h_batch_images = new double [padded_height * padded_width * batch_image_size];
    cudaMalloc(&d_batch_images_spectrum, batch_image_size * padded_height * (padded_width/2 + 1) * sizeof(dComplex));       GPU_MEM += batch_image_size * padded_height * (padded_width/2 + 1) * sizeof(dComplex);
    h_batch_images_spectrum = new double [2 * batch_image_size * padded_height * (padded_width/2 + 1)];

    // iii) for convoluted spectrum and image (same as # of template, shared by every image
    h_mul_spectrum = new double [2 * num_template * padded_height * (padded_width/2 + 1)];
    cudaMalloc(&d_mul_spectrum, num_template * padded_height * (padded_width/2 + 1) * sizeof(dComplex)); GPU_MEM += num_template * padded_height * (padded_width/2 + 1) * sizeof(dComplex);
    cudaMalloc(&d_convolved, num_template * padded_height * padded_width * sizeof(double));               GPU_MEM += num_template * padded_height * padded_width * sizeof(double);
    h_convolved = new double [num_template * padded_height * padded_width];

    // iv) for scoremap
    h_scoreMap = new double[num_template * padded_height * padded_width];
    cudaMalloc(&d_scoreMap, num_template * padded_height * padded_width * sizeof(double));               GPU_MEM += num_template * padded_height * padded_width * sizeof(double);

    // v) for result
    cudaMalloc(&d_minval,   num_template * sizeof(double));      GPU_MEM += num_template * sizeof(double);
    cudaMalloc(&d_argmin_x, num_template * sizeof(int));        GPU_MEM += num_template * sizeof(int);
    cudaMalloc(&d_argmin_y, num_template * sizeof(int));        GPU_MEM += num_template * sizeof(int);
    h_minval = new double[num_template];
    h_argmin_y = new int[num_template];
    h_argmin_x = new int[num_template];

    cufftPlan2d(&fftPlanFwd, padded_height, padded_width, CUFFT_D2Z); // note the order here !!!!!
    cufftPlan2d(&fftPlanInv, padded_height, padded_width, CUFFT_Z2D);

    printf("Allocate %f MBytes\n", GPU_MEM/1e6);
}

Worker::~Worker(){
    delete[] h_templ_sqsum;
    cudaFree(d_templ_sqsum);
    cudaFree(d_templ_height);
    cudaFree(d_templ_width);
    cudaFree(d_templ_sqsum);
    delete[] h_templ_height;
    delete[] h_templ_width;
    cudaFree(d_templates);
    delete[] h_templates;

    //delete[] h_batch_images;
    //cudaFree(d_scores);
    //cudaFree(d_convoluted); 
    //cudaFree(d_integral_img);

    cufftDestroy(fftPlanFwd);
    cufftDestroy(fftPlanInv);
}

void Worker::run(){
    //test();
    // for(int gpu = 0; gpu < 1; gpu++){
        makeTemplateReady(); // load template, keep sq sum, keep spectrum
        makeImageReady(); //keep spectrum, then keep sqIntegral image (in place) 
        matchTemplate();
    // }
}

void Worker::makeTemplateReady(){
    // Goal: keep sq sum, keep spectrum
    // read templates
    ifstream readtemplist;
    readtemplist.open(template_list.c_str());
    string output;
    vector<string> temp;
    if (readtemplist.is_open()) {
        while (getline (readtemplist, output)) {
            temp.push_back(output);
        }
    }
    readtemplist.close();
    endTime = CycleTimer::currentSeconds();
    printf("Time for loading template string: %f sec.\n", endTime - startTime);
    fflush(stdout);

    // read templates' sizes
    printf("number of template: %d\n", num_template);
    int c = 0, max_width = 0, max_height = 0;
    for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
        string temp_name = "/" + *iter;
        Mat templ = imread(template_folder + "/" + (*iter), 0); //gray-scale image!
        h_templ_width[c] = templ.cols;
        h_templ_height[c] = templ.rows;
        if(templ.cols > max_width) max_width = templ.cols;
        if(templ.rows > max_height) max_height = templ.rows;
        //printf("template %d: %dx%d\n", c, templ.cols, templ.rows);
        c++;
    }
    cudaMemcpy(d_templ_width, h_templ_width, num_template*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_templ_height, h_templ_height, num_template*sizeof(int), cudaMemcpyHostToDevice);
    printf("Maximum template size: %dx%d\n", max_width, max_height);

    // load template and save fft
    //int pitch2 = padded_width * padded_height;
    int pitch2_half = padded_height * (padded_width/2 + 1);
    int pitch1 = padded_width;
    c = 0;
    
    startTime = CycleTimer::currentSeconds();
    for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
        string temp_name = "/" + *iter;
        if(c == 14) {printf("Processing template %d, %s\n", c, temp_name.c_str()); fflush(stdout);}
        Mat templ = imread(template_folder + "/" + (*iter), CV_LOAD_IMAGE_COLOR); //gray-scale image!
        double sqsum = 0;
        //printf("Processing template %d before for loop\n", c);
        //note we have to reverse the template for convolution
        for(int j = 0; j < padded_height; j++){
            for(int i = 0; i < padded_width; i++){
                if(j < templ.rows && i < templ.cols){
                    Vec3b pixel = templ.at<Vec3b>(Point(i,j));
                    double gpixel = floor(pixel.val[0] * .114 + pixel.val[1] * .587 + pixel.val[2] * .299 + 1e-5);
                    //if(c == 14) printf("%.0f ", gpixel);
                    int j_inv = templ.rows - j - 1;
                    int i_inv = templ.cols - i - 1;
                    h_templates[(j_inv)*pitch1 + i_inv] = gpixel;
                    sqsum += gpixel * gpixel;
                }
                else {
                    h_templates[j*pitch1 + i] = 0;
                }
            }
            //if(c == 14 && j < templ.rows) printf("\n");
        }
        h_templ_sqsum[c] = sqsum;

        // if(c == 14) {
        //     //templ = imread(template_folder + "/" + (*iter), CV_LOAD_IMAGE_COLOR);
        //     for(int j = 0; j < templ.rows; j++){
        //         for(int i = 0; i < templ.cols; i++){
        //             //Vec3b pixel = templ.at<Vec3b>(Point(i,j));
        //             //double gpixel = floor(pixel.val[0] * .114 + pixel.val[1] * .587 + pixel.val[2] * .299);
        //             printf("%.1f ", h_templates[j*pitch1 + i]);
        //             //printf("%.1f ", gpixel);
        //         }
        //         printf("\n");
        //     }
        //     printf("h_templ_sqsum[%d] = %f;\n", c, h_templ_sqsum[c]);
        // }
        
        //printf("Processing template %d before for memcpy\n", c); fflush(stdout);
        cudaMemcpy(d_templates, h_templates, padded_width * padded_height * sizeof(double), cudaMemcpyHostToDevice);
        //fft
        //printf("Processing template %d before for fft\n", c); fflush(stdout);
        cufftResult result = cufftExecD2Z(fftPlanFwd, (cufftDoubleReal *)d_templates, 
             (cufftDoubleComplex *)(d_all_templates_spectrum + c * pitch2_half * 2));
        assert(result == 0);
        c++;
    }
    cudaMemcpy(d_templ_sqsum, h_templ_sqsum, num_template * sizeof(double), cudaMemcpyHostToDevice);
    endTime = CycleTimer::currentSeconds();
    printf("Template done, avg %f sec per template\n", (endTime - startTime) / num_template);

    // //verify d_all_templates_spectrum
    // cudaMemcpy(h_all_templates_spectrum, d_all_templates_spectrum,
    //            num_template * padded_height * (padded_width/2 + 1) * sizeof(dComplex), cudaMemcpyDeviceToHost);
    // printf("Template spectrum:\n");
    // for(int c = 14; c < 15; c++) {
    //     for(int i = 0; i < pitch2_half; i++){
    //         printf("%f %f ", h_all_templates_spectrum[c*pitch2_half*2 + 2*i], h_all_templates_spectrum[c*pitch2_half*2 + 2*i + 1]);
    //         if(i % (padded_width/2 + 1) == padded_width/2) printf("\n");
    //     }
    //     printf("\n\n");
    // }
}

void Worker::makeImageReady(){
    // Goal: keep spectrum, then keep sqIntegral image (in place) 
    //2) allocate images [w * h * batch_size] 
    startTime = CycleTimer::currentSeconds();
    char filename[200];
    int pitch2 = padded_height * padded_width;
    int pitch2_half = padded_height * (padded_width/2 + 1);
    int pitch1 = padded_width;
    for(int c = 0; c < batch_image_size; c++){
        sprintf(filename, image_name.c_str(), c);
        if(c == 0) {printf("Processing image %d, %s\n", c, filename); fflush(stdout);}
        Mat img = imread(filename, CV_LOAD_IMAGE_COLOR); //gray scale image!
        for(int j = 0; j < img.rows; j++){
            for(int i = 0; i < img.cols; i++){
                if(j < img.rows && i < img.cols){
                    Vec3b pixel = img.at<Vec3b>(Point(i,j));
                    double gpixel = floor(double(pixel.val[0]) * .114 + 
                                          double(pixel.val[1]) * .587 + 
                                          double(pixel.val[2]) * .299 + 1e-5); //numerical issue when r=g=b..
                    h_batch_images[c*pitch2 + j*pitch1 + i] = gpixel;
                }
                else {
                    h_batch_images[c*pitch2 + j*pitch1 + i] = 0;
                }
            }
        }
        // if(c == 0) {
        //     printf("image %d\n", c);
        //     for(int j = 0; j < img.rows; j++){
        //         for(int i = 0; i < img.cols; i++){
        //             printf("%f ", h_batch_images[c*pitch2 + j*pitch1 + i]);
        //         }
        //         printf("\n");
        //     }
        // }
    }
    int size = padded_height * padded_width * batch_image_size;
    cudaMemcpy(d_batch_images, h_batch_images, size * sizeof(double), cudaMemcpyHostToDevice);
    endTime = CycleTimer::currentSeconds();
    printf("Image reading done, %f MBytes copied in %f sec, throughput = %f MB/sec, %f sec/img\n", 
             size * sizeof(double)/double(1e6), endTime - startTime, size/(endTime - startTime)/1e6, 
            (endTime - startTime)/batch_image_size);

    //spectrum
    startTime = CycleTimer::currentSeconds();
    for(int c = 0; c < batch_image_size; c++){
        cufftResult result = cufftExecD2Z(fftPlanFwd, (cufftDoubleReal *)(d_batch_images + c * pitch2), 
             (cufftDoubleComplex *)(d_batch_images_spectrum + c * pitch2_half * 2));
        assert(result == 0);
    }
    endTime = CycleTimer::currentSeconds();
    cudaDeviceSynchronize();
    printf("Image spectrum done in %f sec. Avg %f sec per image\n", (endTime - startTime), 
            (endTime - startTime)/batch_image_size);

    //inspect spectrum
    cudaMemcpy(h_batch_images_spectrum, d_batch_images_spectrum, 
               batch_image_size * padded_height * (padded_width/2 + 1) * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    // printf("Image spectrum:\n");
    // for(int c = 0; c < 1; c++) {
    //     for(int i = 0; i < pitch2_half; i++){
    //         printf("%f %f ", h_batch_images_spectrum[c*pitch2_half*2 + 2*i], h_batch_images_spectrum[c*pitch2_half*2 + 2*i + 1]);
    //         if(i % (padded_width/2 + 1) == padded_width/2) printf("\n");
    //     }
    //     printf("\n\n");
    // }
    
    //integral images in place
    startTime = CycleTimer::currentSeconds();
    int total_rows = image_height * batch_image_size; //300 * 128
    int total_cols = (image_width+1) * batch_image_size;  //(426+1) * 128, note +1 here
    int numBlocks = updiv(total_rows, numThreadsPerBlock);
    printf("Launching sqrIntegralKernelPass1<<<%d, %d>>>\n", numBlocks, numThreadsPerBlock);
    sqrIntegralKernelPass1<<<numBlocks, numThreadsPerBlock>>>(batch_image_size, image_width, image_height, padded_width, padded_height, d_batch_images);

    //size = padded_height * padded_width * batch_image_size;
    //double* h_integral_img_debug = new double[size];
    // cudaMemcpy(h_integral_img_debug, d_batch_images, size * sizeof(double), cudaMemcpyDeviceToHost);

    // printf("integral image phase 1\n");
    // for(int j=0; j < 40; j++){
    //    for(int i=0; i < 40; i++){
    //        printf("%d ", int(h_integral_img_debug[j*padded_width+i])%100);
    //    }
    //    printf("\n");
    // }
    // printf("\n");

    numBlocks = updiv(total_cols, numThreadsPerBlock);
    printf("Launching sqrIntegralKernelPass2<<<%d, %d>>>\n", numBlocks, numThreadsPerBlock);
    sqrIntegralKernelPass2<<<numBlocks, numThreadsPerBlock>>>(batch_image_size, image_width, image_height, padded_width, padded_height, d_batch_images);
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("Integral image done in %f sec. Avg %f sec per image\n", (endTime - startTime), 
            (endTime - startTime)/batch_image_size);
    
    // printf("integral image:\n");
    // cudaMemcpy(h_integral_img_debug, d_batch_images, size * sizeof(double), cudaMemcpyDeviceToHost);
    // for(int j = 300; j < 301; j++){
    //     for(int i = 0; i < 427; i++){
    //         printf("%.1f ", h_integral_img_debug[j*padded_width+i]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}

void Worker::matchTemplate(){
    //given spectrum of image and templates, do dot product, ifft2, and find maximum location & score

    //for each image in batch, get convoluted result
    
    for(int i = 0; i < batch_image_size; i++){
        //startTime = CycleTimer::currentSeconds();
        modulateAndNormalize(i);
        //cudaDeviceSynchronize();
        //endTime = CycleTimer::currentSeconds();
        //printf("matchTemplate done in %f sec.\n", (endTime - startTime));

        //startTime = CycleTimer::currentSeconds();
        getConvolved(i);
        //cudaDeviceSynchronize();
        //endTime = CycleTimer::currentSeconds();
        //printf("getConvolved done in %f sec.\n", (endTime - startTime));

        //startTime = CycleTimer::currentSeconds();
        getScoreMap(i);
        //cudaDeviceSynchronize();
        //endTime = CycleTimer::currentSeconds();
        //printf("getScoreMap done in %f sec.\n", (endTime - startTime));

        //startTime = CycleTimer::currentSeconds();
        findMinLoc(i);
        //cudaDeviceSynchronize();
        //endTime = CycleTimer::currentSeconds();
        //printf("findMinLoc done in %f sec.\n", (endTime - startTime));
        //writeIntoDisc
    }
    
    
    //(endTime - startTime)/batch_image_size);
}

void Worker::modulateAndNormalize(int image_index) {
    //assert(fftW % 2 == 0);
    //d_mul_spectrum = d_all_templates_spectrum(image_index) .* d_batch_images_spectrum
    int pitch1 = padded_width / 2 + 1;
    int pitch2 = padded_height * (padded_width / 2 + 1);
    const int total_rows = num_template * padded_height;
    //printf("total_rows = %d\n", total_rows);

    //printf("Launching modulateAndNormalize_kernel<<<%d, %d>>>\n", updiv(total_rows, numThreadsPerBlock), numThreadsPerBlock);
    //every 32 thread is doing a row, need 32 * total_rows threads
    modulateAndNormalize_kernel<<<updiv(total_rows * 32, numThreadsPerBlock), numThreadsPerBlock>>>
                 ((dComplex*) d_mul_spectrum, 
                  (dComplex*) d_all_templates_spectrum, 
                  (dComplex*) (d_batch_images_spectrum + (2 * pitch2 * image_index)), 
                  total_rows, pitch1, padded_height, double(1.0)/double(padded_height*padded_width));
    //cudaDeviceSynchronize();

    //verify
    // cudaMemcpy(h_mul_spectrum, d_mul_spectrum, num_template * padded_height * (padded_width/2 + 1) * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    
    // printf("spectrum product: \n");
    // for(int c = 14; c < 15; c++){
    //     for(int i = 0; i < pitch2; i++){
    //         printf("%f %f ", h_mul_spectrum[c*pitch2*2 + 2*i], h_mul_spectrum[c*pitch2*2 + 2*i + 1]);
    //         if(i % (padded_width/2 + 1) == padded_width/2) printf("\n");
    //     }
    //     printf("\n");
    // }
}

void Worker::getConvolved(int image_index){
    //d_mul_spectrum           in   num_template * padded_height * (padded_width/2 + 1) (dComplex)
    //d_all_templates_spectrum in   num_template * padded_height * padded_width         (double)
    int pitch2_half = padded_height * (padded_width/2 + 1);
    int pitch2 = padded_height * padded_width;

    for(int t = 0; t < num_template; t++){
        cufftResult result = cufftExecZ2D(fftPlanInv, (cufftDoubleComplex *)(d_mul_spectrum + t * pitch2_half * 2), 
                 (cufftDoubleReal *)(d_convolved + t * pitch2));
        assert(result == 0);
    }

    // //verify
    // printf("convolved: \n");
    // if(image_index == 0){
    //     int pitch1 = padded_width;
    //     cudaMemcpy(h_convolved, d_convolved, num_template * padded_height * padded_width * sizeof(double), cudaMemcpyDeviceToHost);
    //     //printf("convolved result:\n");
    //     for(int c = 14; c < 15; c++){
    //         for(int i = 0; i < padded_height; i++){
    //             for(int j = 0; j < padded_width; j++){
    //                 printf("%f ", h_convolved[c * pitch2 + i * pitch1 + j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
}

void Worker::getScoreMap(int image_index){
    //32 thread per row
    int total_rows = num_template * padded_height;
    //printf("Launching getScoreMap_kernel<<<%d,%d>>>\n", updiv(total_rows * 32, numThreadsPerBlock), numThreadsPerBlock);
    getScoreMap_kernel<<<updiv(total_rows * 32, numThreadsPerBlock), numThreadsPerBlock>>>
                (d_scoreMap, d_batch_images, d_convolved, d_templ_sqsum, d_templ_width, d_templ_height, 
                 num_template, padded_width, padded_height, image_width, image_height, image_index);

    //verify
    //cudaMemcpy(h_scoreMap, d_scoreMap, num_template * padded_height * padded_width * sizeof(double), cudaMemcpyDeviceToHost);
    //int pitch2 = padded_height * padded_width;

    // printf("score map:\n");
    // for(int c = 14; c < 15; c++){
    //     printf("For template %d size is %d x %d\n", c, image_height - h_templ_height[c] + 1, image_width - h_templ_width[c] + 1);
    //     for(int i = 0; i < image_height - h_templ_height[c] + 1; i++){
    //         for(int j = 0; j < image_width - h_templ_width[c] + 1; j++){
    //             if(i >= 146 && i <= 153 && j >= 205 && j <= 209)
    //                 printf("%.3e ", h_scoreMap[c * pitch2 + i * padded_width + j]);
    //         }
    //         if(i >= 146 && i <= 153) 
    //                 printf("\n");
    //     }
    // }
}

void Worker::findMinLoc(int image_index){
    //find minMinLoc and index in d_scoreMap (num_template * padded_width * padded_height with actual smaller size)
    //Phase 1: 32 threads per row for row_reduce
    int total_rows = num_template * image_height; //note we are paying extra efforts on padded region
    //printf("Launching rowReduceMin_kernel<<<%d,%d>>>\n", updiv(total_rows * 32, numThreadsPerBlock), numThreadsPerBlock);
    rowReduceMin_kernel<<<updiv(total_rows * 32, numThreadsPerBlock), numThreadsPerBlock>>>
                (d_scoreMap, d_templ_width, d_templ_height, num_template, 
                 padded_width, padded_height, image_width, image_height);

    //verify
    // cudaMemcpy(h_scoreMap, d_scoreMap, num_template * padded_height * padded_width * sizeof(double), cudaMemcpyDeviceToHost);
    // int pitch2 = padded_height * padded_width;

    // for(int c = 3; c < 4; c++){
    //     printf("For template %d size is %d x %d\n", c, image_height - h_templ_height[c] + 1, image_width - h_templ_width[c] + 1);
    //     for(int i = 0; i < image_height - h_templ_height[c] + 1; i++){
    //         for(int j = 0; j < image_width - h_templ_width[c] + 1; j++){
    //             if(j < 2)
    //             printf("%.3e ", h_scoreMap[c * pitch2 + i * padded_width + j]);
    //         }
    //         printf("\n");
    //     }
    // }
    
    //Phase 2: 32 threads per template for column reduce
    int total_threads = 32 * num_template;
    //printf("Launching columnReduceMin_kernel<<<%d,%d>>>\n", updiv(total_threads, numThreadsPerBlock), numThreadsPerBlock);
    columnReduceMin_kernel<<<updiv(total_threads, numThreadsPerBlock), numThreadsPerBlock>>>
                (d_scoreMap, d_templ_width, d_templ_height, d_minval, d_argmin_x, d_argmin_y,
                 num_template, padded_width, padded_height, image_width, image_height);

    // if(image_index == 400){
    //     //verify
    //     cudaMemcpy(h_minval, d_minval, num_template * sizeof(double), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_argmin_x, d_argmin_x, num_template * sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_argmin_y, d_argmin_y, num_template * sizeof(int), cudaMemcpyDeviceToHost);

    //     for(int i=0;i<num_template;i++){ //num_template;i++){
    //         printf("template %d: score: %f, x: %d, y: %d\n", i, h_minval[i], h_argmin_x[i], h_argmin_y[i]);
    //     }
    // }
}
