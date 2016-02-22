/* This sample demonstrates the way you can perform independed tasks
   on the different GPUs */

// Disable some warnings which are caused with CUDA headers
#if defined(_MSC_VER)
#pragma warning(disable: 4201 4408 4100)
#endif

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

//#define HAVE_TBB
#define HAVE_CUDA

// #ifdef HAVE_TBB
// #  include "tbb/tbb_stddef.h"
// #  if TBB_VERSION_MAJOR*100 + TBB_VERSION_MINOR >= 202
// #    include "tbb/tbb.h"
// #    include "tbb/task.h"
// #    undef min
// #    undef max
// #  else
// #    undef HAVE_TBB
// #  endif
// #endif

// #if !defined(HAVE_CUDA) || !defined(HAVE_TBB)

// int main()
// {
//   #if !defined(HAVE_CUDA)
//       std::cout << "CUDA support is required (CMake key 'WITH_CUDA' must be true).\n";
//   #endif

//   #if !defined(HAVE_TBB)
//       std::cout << "TBB support is required (CMake key 'WITH_TBB' must be true).\n";
//   #endif

//   return 0;
// }

// #else

using namespace std;
using namespace cv;
using namespace cv::gpu;

vector<float> score;
vector<Point> position;
vector<string> template_name;

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
  if(argc < 10){
    cout << "Usage: ./c1413_notbb [device_id] [device_order] [total_device] [template_list] [template_folder] [image_name] [result_filename] [begin_image_num] [end_image_num]\n";
    exit(1);
  }

  //check GPUs
  int num_devices = getCudaEnabledDeviceCount();
  if (num_devices < 1) {
    cout << "At least one GPU is required\n";
    return -1;
  }

  for (int i = 0; i < num_devices; ++i) {
    printShortCudaDeviceInfo(i);

    DeviceInfo dev_info(i);
    if (!dev_info.isCompatible()) {
      cout << "CUDA module isn't built for GPU #" << i << " ("
           << dev_info.name() << ", CC " << dev_info.majorVersion()
           << dev_info.minorVersion() << "\n";
      return -1;
    }
  }

  size_t time = clock();

  //allocate global memories
  //score.resize(device_id.size());
  //position.resize(device_id.size());
  int num_template = 96; // hard coded now. should be dynamically determined by looking into template file list
  //for(int i=0; i<device_id.size(); i++){
  score.resize(num_template);
  position.resize(num_template);
  //}
  template_name.resize(num_template);

  //tbb::parallel_do(devices, devices + 2, Worker());
  Worker w;
  cout << "before setup" << endl;
  w.setupWorker(argv);
  cout << "before run" << endl;
  w.run();

  printf("GPU runtime: %f s\n", (double(clock() - time)/CLOCKS_PER_SEC));
  return 0;
}


void Worker::run() const {
  cout << "run: using GPU " << device_id << endl;
  setDevice(device_id);  	
  ifstream readtemplist;
	readtemplist.open(template_list.c_str());
	string output;
	vector<string> temp;
  string temp_name;

	if (readtemplist.is_open()) {
	  while (getline (readtemplist, output)) {
  		temp.push_back(output);
  		//cout << output << endl;
		}
	}
  printf("Device %d: templates are read. \n", device_id);
	readtemplist.close();
	
	Mat img, templ, result, img_display;
	gpu::GpuMat d_temp[96];
  gpu::GpuMat d_img, d_score;
  
  //upload all templates
  int c = 0;
  for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
    temp_name = "/" + *iter;
    templ = imread(template_folder + "/" + (*iter), 0);
    //printf("%d x %d\n", templ.cols, templ.rows);
    d_temp[c].upload(templ);
    c++;
  }

	double minVal; 
	Point minLoc;
	char filename[200];

	size_t time = clock();

  int numAll = (end_image_num - begin_image_num) + 1;
  cout << end_image_num << " " << begin_image_num << "numAll: " << numAll << " num_devices: " << num_devices << endl;
  int share =  numAll / num_devices;
  if(device_order < (numAll % num_devices)) {
    share++;
  }
  printf("Device order %d (%d): share is %d\n", device_order, device_id, share);

  for(int i=0; i<share; i++){
		int num = begin_image_num + device_order + i * num_devices;
		sprintf(filename, image_name.c_str(), num);
		img = imread(filename, 0);
    size_t time1 = clock();
    //printf("%s: %d x %d\n", filename, img.cols, img.rows);
		d_img.upload(img);
    printf("Capture upload time: %s %f\n", filename, double(clock() - time1)/CLOCKS_PER_SEC);

    size_t time2 = clock();
		int count = 0;
  	for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
      //temp_name = "/" + *iter;
      //templ = imread(template_folder + "/" + (*iter), 0);
      //printf("%d x %d\n", templ.cols, templ.rows);
    	//d_temp.upload(templ);
    	//cout << num << "  "<< count << endl;

      gpu::matchTemplate(d_img, d_temp[count], d_score, CV_TM_SQDIFF_NORMED);
      gpu::minMaxLoc(d_score, &minVal, NULL, &minLoc, NULL);

      //printf("%d,%d \n",device_id,count);
      score[count]= minVal;
      position[count]= minLoc;
      template_name[count] = temp_name;
      count++;
   	}
    printf("Processing Time: %f\n", double(clock() - time1)/CLOCKS_PER_SEC);
	   
    size_t time3 = clock();
    fstream filea;
    sprintf(filename, result_filename.c_str(), num);
    filea.open(filename, ios::out);
    for (int t=0; t<96; t++){
      filea << score[t] << "\t" 
            << position[t].x << "\t"
            << position[t].y << "\t" 
            << template_name[t] << endl;
    }
    printf("Write Time: %f\n", double(clock() - time3)/CLOCKS_PER_SEC);

    double secs = (double(clock() - time)/CLOCKS_PER_SEC);
    double done_fraction = float(i+1)/share;
    double time_needed = secs/(done_fraction) * (1 - done_fraction) / 3600;
    cout << i << "/" << share << " = " << done_fraction << " \t has run " << secs/3600 << " hrs" 
         << " still need: " << time_needed << " hours"<< endl; 
  }
  // Deallocate data here, otherwise deallocation will be performed
  // after context is extracted from the stack
  d_img.release();
  for(int i=0;i<96;i++){
    d_temp[i].release();
  }
  d_score.release();
}

//#endif
