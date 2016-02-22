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

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/gpu/gpu.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
//#define HAVE_TBB
#define HAVE_CUDA

#include "common/CycleTimer.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

vector<float> score;
vector<Point> position;
vector<string> template_name;
char color_string[] = "\x1b[31m";

struct Worker { 
  string template_list;
  string template_folder;
  string image_name;
  string result_filename;
  int begin_image_num;
  int end_image_num;
  int device_id;
  int num_devices;
  int device_order;
  void run() const;
  void setupWorker(char *argv[]);
};

string type2str(int type);

void Worker::setupWorker(char *argv[]) {
  //./c1413_notbb 1[device_id] 2[device_order] 3[total_device] 4[template_list] 5[template_folder] 
  //              6[image_name] 7[result_filename] 8[begin_image_num] 9[end_image_num]
  // for(int i=0;i<10;i++){
  //   cout << "argv[" << i << "]" << " = " << argv[i] << endl;
  // }

  template_list = argv[4];
  template_folder = argv[5];
  image_name = argv[6];
  result_filename = argv[7];

  begin_image_num = atoi(argv[8]);
  end_image_num = atoi(argv[9]);
  device_id = atoi(argv[1]);
  num_devices = atoi(argv[3]);
  device_order = atoi(argv[2]);
}

int main(int argc, char *argv[]) {

  if(argc < 11){
    cout << "Usage: ./c1413_notbb [device_id] [device_order] [total_device] [template_list] [template_folder] [image_name] [result_filename] [begin_image_num] [end_image_num] [color]\n";
    exit(1);
  }
  
  if(!strcmp(argv[10], "red")){
    strcpy(color_string, "\x1b[31m");
  }
  else if(!strcmp(argv[10], "green")){
    strcpy(color_string, "\x1b[32m");
  }
  else if(!strcmp(argv[10], "yellow")){
    strcpy(color_string, "\x1b[33m");
  }
  else {
    strcpy(color_string, "\x1b[34m");
  }
  

  //check GPUs
  double startTime;// = CycleTimer::currentSeconds();

  // int num_devices = getCudaEnabledDeviceCount();
  // if (num_devices < 1) {
  //   cout << "At least one GPU is required\n";
  //   return -1;
  // }

  // for (int i = 0; i < num_devices; ++i) {
  //   printShortCudaDeviceInfo(i);

  //   DeviceInfo dev_info(i);
  //   if (!dev_info.isCompatible()) {
  //     cout << "CUDA module isn't built for GPU #" << i << " ("
  //          << dev_info.name() << ", CC " << dev_info.majorVersion()
  //          << dev_info.minorVersion() << "\n";
  //     return -1;
  //   }
  // }
  double endTime;// = CycleTimer::currentSeconds();
  //printf("%s""Time for checking GPU: %f sec." ANSI_COLOR_RESET "\n", color_string, endTime - startTime);
  // size_t time = clock();

  //allocate global CPM memories
  startTime = CycleTimer::currentSeconds();
  int num_template = 96; // hard coded now. should be dynamically determined by looking into template file list
  score.resize(num_template);
  position.resize(num_template);
  template_name.resize(num_template);
  //cudaDeviceSynchronize();
  endTime = CycleTimer::currentSeconds();
  printf("%sTime for allocating CPU global memory: %f sec." ANSI_COLOR_RESET "\n", color_string, endTime - startTime);

  // //tbb::parallel_do(devices, devices + 2, Worker());
  Worker w;
  //cout << "before setup" << endl;
  w.setupWorker(argv);
  //cout << "before run" << endl;
  w.run();

  // printf("GPU runtime: %f s\n", (double(clock() - time)/CLOCKS_PER_SEC));
  return 0;
}


void Worker::run() const {
  
  //printf("%sGot in run()\n" ANSI_COLOR_RESET "\n", color_string);
  double startTime = CycleTimer::currentSeconds();
  cudaSetDevice(device_id); //speed reply on system memory?
  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();
  printf("%sTime for setID: %f sec." ANSI_COLOR_RESET "\n", color_string, endTime-startTime);
  fflush(stdout);


  startTime = CycleTimer::currentSeconds();
  printf("%srun: using GPU %d" ANSI_COLOR_RESET "\n", color_string, device_id);
  ifstream readtemplist;
	readtemplist.open(template_list.c_str());
	string output;
	vector<string> temp;
  string temp_name;
	if (readtemplist.is_open()) {
	  while (getline (readtemplist, output)) {
  		temp.push_back(output);
		}
	}
	readtemplist.close();
  //cudaDeviceSynchronize();//actually not necessary because no gpu activity here
  endTime = CycleTimer::currentSeconds();
  printf("%sTime for loading template string: %f sec." ANSI_COLOR_RESET "\n", color_string, endTime - startTime);
  fflush(stdout);


  //allocating memory for templates
  startTime = CycleTimer::currentSeconds();
	//Mat img;
  Mat result, img_display;
  Mat templ;
	gpu::GpuMat d_temp[96];
  CudaMem d_img_pl(300, 426, CV_8UC1, CudaMem::ALLOC_PAGE_LOCKED);
  gpu::GpuMat d_img, d_score;
  endTime = CycleTimer::currentSeconds();
  printf("%sTime for allocating GPU memory: %f sec." ANSI_COLOR_RESET "\n", color_string, endTime - startTime);
  fflush(stdout);


  //upload all templates
  startTime = CycleTimer::currentSeconds();
  int c = 0;
  for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
    temp_name = "/" + *iter;
    templ = imread(template_folder + "/" + (*iter), CV_8UC3);
    d_temp[c].upload(templ);
    c++;
  }
  //cudaDeviceSynchronize(); //not necessary either, because upload is a blocking call
  endTime = CycleTimer::currentSeconds();
  printf("%sTime for uploading templates to GPU: %f sec." ANSI_COLOR_RESET "\n", color_string, endTime - startTime);
  fflush(stdout);


  int numAll = (end_image_num - begin_image_num) + 1;
  //cout << end_image_num << " " << begin_image_num << "numAll: " << numAll << " num_devices: " << num_devices << endl;
  int share =  numAll / num_devices;
  if(device_order < (numAll % num_devices)) {
    share++;
  }
  printf("Device order %d (%d): share is %d\n", device_order, device_id, share);
  fflush(stdout);


  //shared variable for each image
  double minVal; 
  Point minLoc;
  char filename[200];


  gpu::Stream stream;

  for(int i=0; i<share; i++){

    startTime = CycleTimer::currentSeconds();
		int num = begin_image_num + device_order + i * num_devices;
		sprintf(filename, image_name.c_str(), num);
    Mat img = d_img_pl;
		img = imread(filename, 0);
    endTime = CycleTimer::currentSeconds();
    printf("%sTime for reading capture from HDD: img#%d: %f" ANSI_COLOR_RESET "\n", color_string, i, endTime - startTime);
    fflush(stdout);


    startTime = CycleTimer::currentSeconds();
		//d_img.upload(img, stream);
    stream.enqueueUpload(d_img_pl, d_img);
    //cudaDeviceSynchronize(); not necessary either, because upload is a blocking call
    endTime = CycleTimer::currentSeconds();
    printf("%sTime for uploading capture to GPU: img#%d: %f" ANSI_COLOR_RESET "\n", color_string, i, endTime - startTime);
    fflush(stdout);

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    startTime = CycleTimer::currentSeconds();
		int count = 0;
  	for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
      //cout << type2str(d_img.type()) << type2str(d_temp[count].type()) << endl;
      gpu::matchTemplate(d_img, d_temp[count], d_score, CV_TM_SQDIFF_NORMED, stream);
      //gpu::minMaxLoc(d_score, &minVal, NULL, &minLoc, NULL);
      //score[count]= minVal;
      //position[count]= minLoc;
      //template_name[count] = temp_name;
      count++;
   	}
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    endTime = CycleTimer::currentSeconds();
    printf("%sTime for kernel function on img#%d: %f" ANSI_COLOR_RESET "\n", color_string, i, endTime-startTime);
	   
    // fstream filea;
    // sprintf(filename, result_filename.c_str(), num);
    // filea.open(filename, ios::out);
    // for (int t=0; t<96; t++){
    //   filea << score[t] << "\t" 
    //         << position[t].x << "\t"
    //         << position[t].y << "\t" 
    //         << template_name[t] << endl;
    // }
    // printf("Write Time: %f\n", double(clock() - time3)/CLOCKS_PER_SEC);

    // double secs = (double(clock() - time)/CLOCKS_PER_SEC);
    // double done_fraction = float(i+1)/share;
    // double time_needed = secs/(done_fraction) * (1 - done_fraction) / 3600;
    // cout << i << "/" << share << " = " << done_fraction << " \t has run " << secs/3600 << " hrs" 
    //      << " still need: " << time_needed << " hours"<< endl; 
  }
  printf("DONE.\n");
  // Deallocate data here, otherwise deallocation will be performed
  // after context is extracted from the stack
  d_img.release();
  for(int i=0;i<96;i++){
    d_temp[i].release();
  }
  d_score.release();
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

//#endif
