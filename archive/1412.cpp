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

#define HAVE_TBB
#define HAVE_CUDA

#ifdef HAVE_TBB
#  include "tbb/tbb_stddef.h"
#  if TBB_VERSION_MAJOR*100 + TBB_VERSION_MINOR >= 202
#    include "tbb/tbb.h"
#    include "tbb/task.h"
#    undef min
#    undef max
#  else
#    undef HAVE_TBB
#  endif
#endif

#if !defined(HAVE_CUDA) || !defined(HAVE_TBB)

int main()
{
  #if !defined(HAVE_CUDA)
      std::cout << "CUDA support is required (CMake key 'WITH_CUDA' must be true).\n";
  #endif

  #if !defined(HAVE_TBB)
      std::cout << "TBB support is required (CMake key 'WITH_TBB' must be true).\n";
  #endif

  return 0;
}

#else

using namespace std;
using namespace cv;
using namespace cv::gpu;

float score[2][96];
Point position[2][96];
string template_name[96];


struct Worker { void operator()(int device_id) const; };

int main()
{
  int num_devices = getCudaEnabledDeviceCount();
  if (num_devices < 2)
  {
      std::cout << "Two or more GPUs are required\n";
      return -1;
  }
  for (int i = 0; i < num_devices; ++i)
  {
      cv::gpu::printShortCudaDeviceInfo(i);

      DeviceInfo dev_info(i);
      if (!dev_info.isCompatible())
      {
          std::cout << "CUDA module isn't built for GPU #" << i << " ("
               << dev_info.name() << ", CC " << dev_info.majorVersion()
               << dev_info.minorVersion() << "\n";

          return -1;
      }
  }
  
  size_t time = clock();
  // Execute calculation in two threads using two GPUs
  int devices[] = {0, 1};
  tbb::parallel_do(devices, devices + 2, Worker());

  printf("GPU runtime: %f s\n", (double(clock() - time)/CLOCKS_PER_SEC));
  return 0;
}


void Worker::operator()(int device_id) const
{
  setDevice(device_id);
    	
  ifstream readtemplist;
	readtemplist.open("templates.txt");
	string output;
	vector<string> temp;
        string temp_name;	 

	if (readtemplist.is_open()) {
	  while (getline (readtemplist, output)) {
  		temp.push_back(output);
  		cout<<output<< endl;
		}
	}  
	readtemplist.close();
	
	Mat img; Mat templ; Mat result; Mat img_display;
	gpu::GpuMat d_temp, d_img, d_score;
	//img = imread( "/home/smanian/Screen_Captures/1412-2015-09-04T17-03-57 PNG/1412-2015-09-04T17-03-57_68770.png", 0 );
  	//d_img.upload(img);
  	
	double minVal; 
	Point minLoc;
	char filename[200];

  	size_t time = clock();
        for(int i=0; i<57025; i++){
		int num = 2*i + device_id;
		sprintf(filename,"/home/smanian/Screen_Captures/1412-2015-09-04T17-03-057 PNG/1412-2015-09-04T17-03-57_%05d.png",num);
		img = imread(filename, 0 );	
		d_img.upload(img);

		int count=0;
  	for (vector<string>::iterator iter = temp.begin(); iter != temp.end(); ++iter) {
	temp_name = "/"+ *iter;
        templ = imread( "/home/smanian/Templates/"+ *iter, 0 );
    	d_temp.upload(templ);
    	//cout << num << "  "<< count << endl;

      gpu::matchTemplate( d_img, d_temp, d_score, CV_TM_SQDIFF_NORMED );
      gpu::minMaxLoc( d_score, &minVal, NULL, &minLoc, NULL);

      //printf("%d,%d \n",device_id,count);
      score[device_id][count]= minVal;
      position[device_id][count]= minLoc;
      template_name[count] = temp_name; 
      count++;
   	}
	
    fstream filea;
    sprintf(filename,"./results_desc/1412/result_%05d.txt",num);
    filea.open(filename, ios::out);
    for (int t=0; t<96; t++){
      filea<<score[device_id][t]<<"\t"<<position[device_id][t].x<<"\t"<<
             position[device_id][t].y<< "\t"<< template_name[t] <<endl;
    }
    /*
    fstream fileb;
    sprintf(filename,"./1412_test/1412/position_%05d.txt",num);
    fileb.open(filename, ios::out);
    for (int t=0; t<96; t++) {
      fileb<<position[device_id][t].x<<" "<<position[device_id][t].y<< "\t"<< template_name[t]<<endl;
      }*/
    double secs = (double(clock() - time)/CLOCKS_PER_SEC);
    double done_fraction = float(i+1)/57025;
    double time_needed = secs/(done_fraction) * (1 - done_fraction) / 3600;
    cout << i << "/57025 = " << done_fraction << " \t has run " << secs/3600 << " hrs" 
         << " still need: " << time_needed << " hours"<< endl; 
	}
  // Deallocate data here, otherwise deallocation will be performed
  // after context is extracted from the stack
  d_img.release();
  d_temp.release();
  d_score.release();
}

#endif
