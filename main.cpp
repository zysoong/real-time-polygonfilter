#include "./include/DelaunayTriangulation.h"
#include "./include/EdgeFilter.h"
#include "./include/NodeDetection.h"
#include "./include/NonMaxSuppression.h"
#include "./include/OpenclLoader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/opengl.hpp>
#include <iostream>
#include <fstream>
#include <direct.h>
#include <stdlib.h>
#include <errno.h>
#include <filesystem>
#include <random>


int main(int argc, char** argv)
{

	//--------------------Define initial values-------------------------
	const int NODE_DETECTION_THRESHOLD = 6;
	const int MAX_NODES = 1500;
	//!END----------------Define initial values-------------------------

	//Initialization
	cv::Mat frame;
	cv::VideoCapture cap;

	//select API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API

	// open selected camera using selected API
	cap.open(deviceID, apiID);
	double width = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
	double height = cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);

	// check if video capture succeed
	if (!cap.isOpened()) {
		std::cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

	//Set platform and device for OpenCL
	rp::useOpenCL(0, 0, cv::ocl::Device::TYPE_GPU);

	//compile cl code;
	cv::ocl::Program nmsProgram = rp::compile("NonMaxSuppression.cl");

	//get OpenCL kernels
	cv::ocl::Kernel kNMS("nonMaxSuppression", nmsProgram);

	//define kernel parameters
	size_t globalSize[2] = { (size_t)width, (size_t)height };
	size_t localSize[2] = { 8, 8 };

	// allocate cpu mats
	const cv::Mat black = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

	// allocate gpu mats
	cv::UMat inputU = black.getUMat(cv::ACCESS_RW);
	cv::UMat outputU = black.getUMat(cv::ACCESS_RW);
	cv::UMat delaunayU = black.getUMat(cv::ACCESS_RW);


	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);

		// check if frame succesfully generated from the camera
		if (frame.empty()) {
			std::cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		inputU = frame.getUMat(cv::ACCESS_RW);
		delaunayU = cv::UMat::zeros(cv::Size(width, height), CV_8UC3);

		// apply grayscale convertion
		cv::cvtColor(inputU, outputU, cv::COLOR_BGR2GRAY);
		inputU = outputU;

		// apply box filter
		cv::boxFilter(inputU, outputU, -1, cv::Size(3, 3));
		inputU = outputU;
		imshow("Box filter", outputU);

		// apply edge filter with OpenCL
		rp::edgeFilterISD(inputU, outputU);
		imshow("Edge Before Suppressed", outputU);
		inputU = outputU;

		// apply non max suppression
		kNMS.args(
			cv::ocl::KernelArg::ReadWrite(inputU)
		).run(2, globalSize, localSize, true);
		outputU = inputU;   //manually pass input to output because this is a in-place function
		imshow("Edge After Suppressed", outputU);

		// apply node detection
		std::vector<cv::Point2f> pickedNodes = rp::nodeDetection(inputU, outputU, NODE_DETECTION_THRESHOLD, MAX_NODES);

		// apply delaunay triangulation
		cv::Rect rect(0, 0, outputU.size().width, outputU.size().height);
		cv::Subdiv2D subdiv(rect);
		subdiv = rp::delaunayTriangulation(pickedNodes, subdiv);

		// render result of delaunay triangulation
		rp::drawLines(delaunayU, subdiv);

		// show live and wait for a key with timeout long enough to show images
		imshow("delaunay", delaunayU);
		if (cv::waitKey(5) >= 0)
			break;
	}

	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;

}