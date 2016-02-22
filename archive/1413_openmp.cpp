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
#include <time.h>
#include <omp.h>
#include <pthread.h>

#define HAVE_CUDA

#if !defined(HAVE_CUDA)
int main()
{
  std::cout << "CUDA support is required (CMake key 'WITH_CUDA' must be true).\n";
  return 0;
}
#else

using namespace std;
using namespace cv;
using namespace cv::gpu;

vector<vector<float> > score;
vector<vector<Point> > position;
vector<string> template_name;

class Worker {
  public:
    static void setupWorker(vector<int> device_id);
    void run(int device_order) const;
  private:  
    static string template_list;
    static string template_folder;
    static string image_name;
    static string result_filename;
    static int begin_image_num;
    static int end_image_num;
    static vector<int> device_set;
    static int num_devices;
};

// ----- edit this part ------
string Worker::template_list = "templates.txt";
string Worker::template_folder = "Templates/";
string Worker::image_name = "1413-2015-09-04T17-03-33 PNG/1413-2015-09-04T17-03-33_%05d.png";
string Worker::result_filename = "result/1413/result_%05d.txt";
int Worker::begin_image_num = 0;
int Worker::end_image_num = 63;//114048;
// ----------------------------
vector<int> Worker::device_set = vector<int>();
int Worker::num_devices = 0;

void Worker::setupWorker(vector<int> device_id) {
  device_set = device_id;
  num_devices = device_id.size();
}

int main(int argc, char *argv[]) {
  //check GPUs
  int num_devices = getCudaEnabledDeviceCount();
  if (num_devices < 1) {
    cout << "At least one GPU is required\n";
    return -1;
  }
  else {
    cout << num_devices << " GPU(s) are available.\n";
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

  // select GPUs
  vector<int> device_id;
  device_id.push_back(0);
  device_id.push_back(1);
  //cout << "Using device " << gpu_id_selected << endl;
  Worker::setupWorker(device_id);

  //allocate global memories
  score.resize(device_id.size());
  position.resize(device_id.size());
  int num_template = 96; // hard coded now. should be dynamically determined by looking into template file list
  for(int i=0; i<device_id.size(); i++){
    score[i].resize(num_template);
    position[i].resize(num_template);
  }
  template_name.resize(num_template);

  //run
  vector<int> device_order;
  for(int i=0; i<device_id.size(); i++){
    device_order.push_back(i); 
  }

  cout << "Device list: ";
  for(int i=0; i<device_order.size(); i++){
     cout << device_order[i] << "(" << device_id[i] << ")" << endl;
  }

  struct timespec start, finish;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  
  //tbb::parallel_do(device_order.begin(), device_order.end(), Worker());
  //start openmp
  omp_set_num_threads(2);
  int TID = 0;
  #pragma omp parallel
  {
    TID = omp_get_thread_num();
    cout<<"There are "<<omp_get_num_threads()<<" threads"<<endl;
    cout<<"And this is number "<<TID<<endl;
    Worker w;
    w.run(TID);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

  elapsed = double(finish.tv_sec - start.tv_sec) + double(finish.tv_nsec - start.tv_nsec)/double(1e9);
  printf("GPU runtime: %f s\n", elapsed);
  return 0;
}

void Worker::run(int device_order) const {
  assert(device_order < num_devices);
  setDevice(device_set[device_order]);
      
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
  printf("Device order %d (%d): templates are read. \n", device_order, device_set[device_order]);
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

  int numAll = (end_image_num - begin_image_num) + 1;
  int share =  numAll / num_devices;
  if(device_order < (numAll % num_devices)) {
    share++;
  }
  printf("Device order %d (%d): share is %d\n", device_order, device_set[device_order], share);

  for(int i=0; i<share; i++){
    int num = begin_image_num + device_order + i * num_devices;
    sprintf(filename, image_name.c_str(), num);
    img = imread(filename, 0);
    //size_t time1 = clock();
    //printf("%s: %d x %d\n", filename, img.cols, img.rows);
    d_img.upload(img);
    //printf("Capture upload time: %s %f\n", filename, double(clock() - time1)/CLOCKS_PER_SEC);
    printf("Processing %s\n", filename);
    //size_t time2 = clock();

    struct timespec start, finish;
    //double start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //start = (double)gethrtime() / double(1e9);

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
      // score[device_order][count]= minVal;
      // position[device_order][count]= minLoc;
      // template_name[count] = temp_name;
      count++;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
    elapsed = double(finish.tv_sec - start.tv_sec) + double(finish.tv_nsec - start.tv_nsec)/double(1e9);
    printf("Process time: %f s\n", elapsed);
    //cout << start.tv_sec << "." << start.tv_nsec << "  " << finish.tv_sec << "." << finish.tv_nsec << endl;

    //printf("Processing Time: %f\n", double(clock() - time2)/CLOCKS_PER_SEC);
     
    //size_t time3 = clock();
    fstream filea;
    sprintf(filename, result_filename.c_str(), num);
    filea.open(filename, ios::out);
    for (int t=0; t<96; t++){
      filea << score[device_order][t] << "\t" 
            << position[device_order][t].x << "\t"
            << position[device_order][t].y << "\t" 
            << template_name[t] << endl;
    }
    //printf("Write Time: %f\n", double(clock() - time3)/CLOCKS_PER_SEC);

  
    // double secs = (double(clock() - time)/CLOCKS_PER_SEC);
    // double done_fraction = float(i+1)/share;
    // double time_needed = secs/(done_fraction) * (1 - done_fraction) / 3600;
    // cout << i << "/" << share << " = " << done_fraction << " \t has run " << secs/3600 << " hrs" 
    //      << " still need: " << time_needed << " hours"<< endl; 
  }
  // Deallocate data here, otherwise deallocation will be performed
  // after context is extracted from the stack
  d_img.release();
  for(int i=0;i<96;i++){
    d_temp[i].release();
  }
  d_score.release();
}

#endif
