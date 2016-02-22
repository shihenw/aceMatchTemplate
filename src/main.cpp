/* This sample demonstrates the way you can perform independent tasks
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
// #if defined(_MSC_VER)
// #pragma warning(disable: 4201 4408 4100)
// #endif
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#include <iostream>
#include <vector>
#include <stdio.h>
#include <string>

#include "util/CycleTimer.h"
#include "worker.h"

using namespace std;

string type2str(int type);
void printCudaInfo();

int main(int argc, char *argv[]) {

    if(argc < 11){
        cout << "Usage: ./c1413_notbb [device_id] [device_order] [total_device] [template_list] [template_folder] [image_name] [result_filename] [begin_image_num] [end_image_num] [color]\n";
        exit(1);
    }

    // if(!strcmp(argv[10], "red")){
    //     strcpy(color_string, "\x1b[31m");
    // }
    // else if(!strcmp(argv[10], "green")){
    //     strcpy(color_string, "\x1b[32m");
    // }
    // else if(!strcmp(argv[10], "yellow")){
    //     strcpy(color_string, "\x1b[33m");
    // }
    // else {
    //     strcpy(color_string, "\x1b[34m");
    // }
    
    double startTime, endTime;// = CycleTimer::currentSeconds();
    //check GPUs
    printCudaInfo();
    
    //printf("Time for allocating CPU global memory: %f sec.\n", color_string, endTime - startTime);

    startTime = CycleTimer::currentSeconds();
    Worker w(argv);
    endTime = CycleTimer::currentSeconds();
    printf("Time: Worker constructor: %f s\n", (endTime - startTime));
    
    printf("0 is %s\n", type2str(0).c_str());

    startTime = CycleTimer::currentSeconds();
    w.run();
    endTime = CycleTimer::currentSeconds();
    printf("Time: Worker run: %f s, average %f per image.\n", (endTime - startTime), (endTime - startTime)/w.get_batch_size());

    return 0;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void printCudaInfo() {
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}