#include <fstream>
#include <iostream>
#include <stdio.h>
#include "worker.h"
#include "util/CycleTimer.h"
#include "myOwnMatchTemplate.cuh"

#define numThreadsPerBlock 256

typedef float2 fComplex;

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
    float GPU_MEM = 0;
    // i) for template
    h_templ_width = new int [num_template];
    h_templ_height = new int [num_template];
    cudaMalloc(&d_templ_width, num_template * sizeof(int));         GPU_MEM += num_template * sizeof(int);
    cudaMalloc(&d_templ_height, num_template * sizeof(int));        GPU_MEM += num_template * sizeof(int);
    h_templ_sqsum = new float [num_template];
    cudaMalloc(&d_templ_sqsum, num_template * sizeof(float));       GPU_MEM += num_template * sizeof(float);
    h_templates = new float [padded_height * padded_width];
    // temp space
    cudaMalloc(&d_templates, padded_height * padded_width * sizeof(float));    GPU_MEM += padded_height * padded_width * sizeof(float);
    h_all_templates_spectrum = new float [2 * num_template * padded_height * (padded_width/2 + 1)];
    cudaMalloc(&d_all_templates_spectrum, num_template * padded_height * (padded_width/2 + 1) * sizeof(fComplex));
                                                                                GPU_MEM += num_template * padded_height * (padded_width/2 + 1) * sizeof(fComplex);
    // ii) for image
    cudaMalloc(&d_batch_images, padded_height * padded_width * batch_image_size * sizeof(float));       GPU_MEM += padded_height * padded_width * batch_image_size * sizeof(float);
    h_batch_images = new float [padded_height * padded_width * batch_image_size];
    cudaMalloc(&d_batch_images_spectrum, batch_image_size * padded_height * (padded_width/2 + 1) * sizeof(fComplex));       GPU_MEM += batch_image_size * padded_height * (padded_width/2 + 1) * sizeof(fComplex);
    h_batch_images_spectrum = new float [2 * batch_image_size * padded_height * (padded_width/2 + 1)];

    // iii) for convoluted spectrum and image (same as # of template, shared by every image
    h_mul_spectrum = new float [2 * num_template * padded_height * (padded_width/2 + 1)];
    cudaMalloc(&d_mul_spectrum, num_template * padded_height * (padded_width/2 + 1) * sizeof(fComplex)); GPU_MEM += num_template * padded_height * (padded_width/2 + 1) * sizeof(fComplex);
    cudaMalloc(&d_convolved, num_template * padded_height * padded_width * sizeof(float));               GPU_MEM += num_template * padded_height * padded_width * sizeof(float);
    h_convolved = new float [num_template * padded_height * padded_width];

    cufftPlan2d(&fftPlanFwd, padded_height, padded_width, CUFFT_R2C); // note the order here !!!!!
    cufftPlan2d(&fftPlanInv, padded_height, padded_width, CUFFT_C2R);

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
        //if(c < 2) {printf("Processing template %d, %s\n", c, temp_name.c_str()); fflush(stdout);}
        Mat templ = imread(template_folder + "/" + (*iter), 0); //gray-scale image!
        float sqsum = 0;
        //printf("Processing template %d before for loop\n", c);
        for(int j = 0; j < padded_height; j++){
            for(int i = 0; i < padded_width; i++){
                if(j < templ.rows && i < templ.cols){
                    float pixel = (float)templ.at<uchar>(Point(i,j));
                    h_templates[j*pitch1 + i] = pixel;
                    sqsum += pixel * pixel;
                }
                else {
                    h_templates[j*pitch1 + i] = 0;
                }
            }
        }
        h_templ_sqsum[c] = sqsum;

        // if(c < 2) {
        //     templ = imread(template_folder + "/" + (*iter), 0);
        //     for(int j = 0; j < templ.rows; j++){
        //         for(int i = 0; i < templ.cols; i++){
        //             printf("%d ", templ.at<uchar>(Point(i,j)));
        //         }
        //         printf("\n");
        //     }
        // }
        
        //printf("Processing template %d before for memcpy\n", c); fflush(stdout);
        cudaMemcpy(d_templates, h_templates, padded_width * padded_height * sizeof(float), cudaMemcpyHostToDevice);
        //fft
        //printf("Processing template %d before for fft\n", c); fflush(stdout);
        cufftResult result = cufftExecR2C(fftPlanFwd, (cufftReal *)d_templates, 
             (cufftComplex *)(d_all_templates_spectrum + c * pitch2_half * 2));
        assert(result == 0);
        c++;
    }
    cudaMemcpy(d_templ_sqsum, h_templ_sqsum, num_template * sizeof(float), cudaMemcpyHostToDevice);
    endTime = CycleTimer::currentSeconds();
    printf("Template done, avg %f sec per template\n", (endTime - startTime) / num_template);

    //verify d_all_templates_spectrum
    // cudaMemcpy(h_all_templates_spectrum, d_all_templates_spectrum,
    //            num_template * padded_height * (padded_width/2 + 1) * sizeof(fComplex), cudaMemcpyDeviceToHost);
    // printf("Template spectrum:\n");
    // for(int c = 0; c < 1; c++) {
    //     for(int i = 0; i < (padded_width/2 + 1)*2; i++){
    //         printf("%.4e ", h_all_templates_spectrum[c*pitch2_half*2 + i]);
    //         //if(i % 2) printf(" | ");
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
        Mat img = imread(filename, 0); //gray scale image!
        for(int j = 0; j < img.rows; j++){
            for(int i = 0; i < img.cols; i++){
                if(j < img.rows && i < img.cols){
                    h_batch_images[c*pitch2 + j*pitch1 + i] = (float)img.at<uchar>(Point(i,j));
                }
                else {
                    h_batch_images[c*pitch2 + j*pitch1 + i] = 0;
                }
            }
        }
        // if(c < 2) {
        //     for(int j = 0; j < 3; j++){
        //         for(int i = 0; i < 40; i++){
        //             printf("%d ", img.at<uchar>(Point(i,j)));
        //         }
        //         printf("\n");
        //     }
        // }
    }
    int size = padded_height * padded_width * batch_image_size;
    cudaMemcpy(d_batch_images, h_batch_images, size * sizeof(float), cudaMemcpyHostToDevice);
    endTime = CycleTimer::currentSeconds();
    printf("Image reading done, %f MBytes copied in %f sec, throughput = %f MB/sec, %f sec/img\n", 
             size * sizeof(float)/float(1e6), endTime - startTime, size/(endTime - startTime)/1e6, 
            (endTime - startTime)/batch_image_size);

    //spectrum
    startTime = CycleTimer::currentSeconds();
    for(int c = 0; c < batch_image_size; c++){
        cufftResult result = cufftExecR2C(fftPlanFwd, (cufftReal *)(d_batch_images + c * pitch2), 
             (cufftComplex *)(d_batch_images_spectrum + c * pitch2_half * 2));
        assert(result == 0);
    }
    endTime = CycleTimer::currentSeconds();
    cudaDeviceSynchronize();
    printf("Image spectrum done in %f sec. Avg %f sec per image\n", (endTime - startTime), 
            (endTime - startTime)/batch_image_size);

    //inspect spectrum
    // cudaMemcpy(h_batch_images_spectrum, d_batch_images_spectrum, 
    //            batch_image_size * padded_height * (padded_width/2 + 1) * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("Image spectrum:\n");
    // for(int c = 1; c < 2; c++) {
    //     for(int i = 0; i < (padded_width/2 + 1)*2; i++){
    //         printf("%.3e ", h_batch_images_spectrum[c*pitch2_half*2 + i]);
    //         //if(i % 2) printf(" | ");
    //     }
    //     printf("\n\n");
    // }
    
    //integral images in place
    startTime = CycleTimer::currentSeconds();
    int total_rows = image_height * batch_image_size; //300 * 128
    int total_cols = image_width * batch_image_size;  //426 * 128
    int numBlocks = updiv(total_rows, numThreadsPerBlock);
    printf("Launching sqrIntegralKernelPass1<<<%d, %d>>>\n", numBlocks, numThreadsPerBlock);
    sqrIntegralKernelPass1<<<numBlocks, numThreadsPerBlock>>>(batch_image_size, image_width, image_height, padded_width, padded_height, d_batch_images);

    // size = padded_height * padded_width * batch_image_size;
    // float* h_integral_img_debug = new float[size];
    // cudaMemcpy(h_integral_img_debug, d_batch_images, size * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int j=0; j < 10; j++){
    //    for(int i=0; i < 1024; i++){
    //        printf("%d ", int(h_integral_img_debug[j*padded_width+i])%10);
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
    // cudaMemcpy(h_integral_img_debug, d_batch_images, size * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int j=0; j < 10; j++){
    //     for(int i=0; i < 1024; i++){
    //         printf("%d ", int(h_integral_img_debug[j*padded_width+i])%10);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}

void Worker::matchTemplate(){
    //given spectrum of image and templates, do dot product, ifft2, and find maximum location & score

    //for each image in batch, get convoluted result
    for(int i = 0; i < 1; i++){
        modulateAndNormalize(i);
        getConvolved(i);
    }
}

void Worker::modulateAndNormalize(int image_index) {
    //assert(fftW % 2 == 0);
    //d_mul_spectrum = d_all_templates_spectrum(image_index) .* d_batch_images_spectrum
    int pitch1 = padded_width / 2 + 1;
    int pitch2 = padded_height * (padded_width / 2 + 1);
    const int total_rows = num_template * padded_height;
    //printf("total_rows = %d\n", total_rows);

    //printf("Launching modulateAndNormalize_kernel<<<%d, %d>>>\n", updiv(total_rows, numThreadsPerBlock), numThreadsPerBlock);
    modulateAndNormalize_kernel<<<updiv(total_rows, numThreadsPerBlock), numThreadsPerBlock>>>
                 ((fComplex*) d_mul_spectrum, 
                  (fComplex*) d_all_templates_spectrum, 
                  (fComplex*) (d_batch_images_spectrum + (2 * pitch2 * image_index)), 
                  total_rows, pitch1, 1.0f / (float)(padded_height * padded_width));
    //cudaDeviceSynchronize();

    // //verify
    //cudaMemcpy(h_mul_spectrum, d_mul_spectrum, num_template * padded_height * (padded_width/2 + 1) * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // printf("spectrum product (first line):\n");
    // for(int c = 0; c < 1; c++){
    //     for(int i = 0; i < padded_height * (padded_width/2 + 1) * 2; i++){
    //         printf("%.3e ", h_mul_spectrum[c * padded_height * (padded_width / 2 + 1) * 2 + i]);
    //     }
    //     printf("\n");
    // }
}

void Worker::getConvolved(int image_index){
    //d_mul_spectrum           in   num_template * padded_height * (padded_width/2 + 1) (fComplex)
    //d_all_templates_spectrum in   num_template * padded_height * padded_width         (float)
    int pitch2_half = padded_height * (padded_width/2 + 1);
    int pitch2 = padded_height * padded_width;
    //int pitch1 = padded_width;

    for(int t = 0; t < 1; t++){
        cufftResult result = cufftExecC2R(fftPlanInv, (cufftComplex *)(d_mul_spectrum + t * pitch2_half * 2), 
                 (cufftReal *)(d_convolved + t * pitch2));
        assert(result == 0);
    }

    // verify
    // printf("convolved result:\n");
    // cudaMemcpy(h_convolved, d_convolved, num_template * padded_height * padded_width * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int c = 0; c < 1; c++){
    //     for(int i = 0; i < 20; i++){
    //         for(int j = 0; j < 10; j++){
    //             printf("%.4e ", h_convolved[c * pitch2 + i * pitch1 + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}


