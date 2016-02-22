#ifndef WORKER_H
#define WORKER_H

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <string>
using namespace std;
using namespace cv;

class Worker {
	public:
		Worker(char *argv[]);
		~Worker();
		//Worker();
		//void setupWorker(char *argv[]);
		void run();

	private:
		void makeTemplateReady(); //sync call!
		void makeImageReady();
	    
	    vector<float> score;
		vector<Point> position;
		vector<string> template_name;
		//char color_string[] = "\x1b[31m";

		//constants
		const int num_template = 96;// hard coded now. should be dynamically determined by looking into template file list
		const int padded_width = 1024; // should be larger than image + template - 1
		const int padded_height = 512;
		const int batch_image_size = 128;
		const int image_width = 426;
	    const int image_height = 300;

		//things to be set by setupWorker()
		string template_list;
	    string template_folder;
	    string image_name;
	    string result_filename;
	    int begin_image_num;
	    int end_image_num;
	    int device_id;
	    int num_devices;
	    int device_order;

	    //data pointers
	    int* h_templ_width;     		int* d_templ_width;
	    int* h_templ_height;			int* d_templ_height;
	    float* h_templates;				float* d_templates; // temp space
	    float* h_templ_sqsum;			float* d_templ_sqsum;
	    //float* h_all_templates_spectrum;
	    								float* d_all_templates_spectrum;
	   	
	   	float* h_batch_images;			float* d_batch_images;
	   	float* h_batch_images_spectrum;	float* d_batch_images_spectrum;


	   	//fftplans
	   	cufftHandle fftPlanFwd;
	   	cufftHandle fftPlanInv;

	    // timer
	    double startTime;
	    double endTime;
};

#endif