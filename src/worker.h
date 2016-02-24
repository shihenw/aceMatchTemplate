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
		int get_batch_size() {return batch_image_size;}

	private:
		void makeTemplateReady(); //sync call!
		void makeImageReady();

		void matchTemplate();
		void modulateAndNormalize(int image_index);
		void getConvolved(int image_index);
		void getScoreMap(int image_index);
		void findMinLoc(int image_index);
	    
	    vector<float> score;
		vector<Point> position;
		vector<string> template_name;
		//char color_string[] = "\x1b[31m";

		//constants
		const int num_template = 96;// hard coded now. should be dynamically determined by looking into template file list
		const int padded_width = 1024; // should be larger than image + template - 1
		const int padded_height = 512;
		const int image_width = 426;
	    const int image_height = 300;
	    const int batch_image_size = 512;

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
	    int* h_templ_width;     			int* d_templ_width;
	    int* h_templ_height;				int* d_templ_height;
	    double* h_templates;				double* d_templates; // temp space
	    double* h_templ_sqsum;				double* d_templ_sqsum;
	    double* h_all_templates_spectrum;	double* d_all_templates_spectrum;
	   	
	   	double* h_batch_images;				double* d_batch_images;
	   	double* h_batch_images_spectrum;	double* d_batch_images_spectrum;

	   	double* h_mul_spectrum;				double* d_mul_spectrum;
	   	double* h_convolved;				double* d_convolved;

	   	double* h_scoreMap;					double* d_scoreMap;

	   	//results
	   	double* h_minval;					double* d_minval; 
	   	int* h_argmin_x;					int* d_argmin_x;
	   	int* h_argmin_y; 					int* d_argmin_y;

	   	//fftplans
	   	cufftHandle fftPlanFwd;
	   	cufftHandle fftPlanInv;

	    // timer
	    double startTime;
	    double endTime;
};

#endif