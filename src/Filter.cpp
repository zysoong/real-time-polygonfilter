#include "Filter.h"
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
//#include <GLFW/glfw3.h>



namespace fs = std::filesystem;

namespace lib {

    namespace opencl {

		void printPlatformNames(){
			std::vector<cv::ocl::PlatformInfo> plats;
			cv::ocl::getPlatfomsInfo(plats);
			for (int i = 0; i < plats.size(); i++) {
				const cv::ocl::PlatformInfo* platform = &plats[i];
				std::cout << "Platform Name of index " << i << ' = ' << platform->name().c_str() << std::endl;
			}
		}

		cv::ocl::Context useOpenCL(int platformIndex, int deviceIndex, int deviceType) {
			std::vector<cv::ocl::PlatformInfo> plats;
			cv::ocl::getPlatfomsInfo(plats);
			const cv::ocl::PlatformInfo* platinfo = &plats[platformIndex];
			std::cout << "Platform Name:" << platinfo->name().c_str() << std::endl;
			cv::ocl::Device c_dev;
			platinfo->getDevice(c_dev, deviceIndex); 
			std::cout << "Device name:" << c_dev.name().c_str() << std::endl;
			cv::ocl::setUseOpenCL(true);
			cv::ocl::Context context;
			context.create(deviceType);
			return context;
		}

		cv::ocl::Program compile(std::string clFile) {

			//Load OpenCL kernel source
			std::ifstream file(fs::current_path() / "src" / clFile);
			if (file.fail()) {
				std::cerr << "Can not find OpenCL kernel " << fs::current_path() / "src" / clFile << std::endl;
			}
			std::string kernelSource((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
			std::cout << "OpenCL program source " << clFile << " loaded and will be compiled: " << std::endl;
			std::cout << "======================================================================================================" << std::endl;
			std::cout << kernelSource << std::endl;
			std::cout << "======================================================================================================" << std::endl;
			cv::ocl::ProgramSource programSource(kernelSource);

			//compile
			cv::String errmsg;
			cv::ocl::Program program(programSource, "", errmsg);
			if (program.ptr() == NULL) {
				std::cerr << "Can't compile OpenCL program:" << std::endl << errmsg << std::endl;
			}
			if (!errmsg.empty()){
				std::cout << "OpenCL program build log:" << std::endl << errmsg << std::endl;
			}
			return program;
		}

		void edgeFilterISD(cv::UMat src, cv::UMat dst) {
			cv::Mat kernelMat = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
			cv::UMat kernelUMat = kernelMat.getUMat(cv::ACCESS_READ);
			cv::filter2D(src, dst, -1, kernelUMat);
		}

    }

    void edgeFilterISD(cv::InputArray src, cv::OutputArray dst) {

        //Kernel of the isotroph detector
        cv::Mat kernelMat = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

        //2D convolution
        cv::filter2D(src, dst, -1, kernelMat);
    }

    void nonMaxSuppresion(cv::Mat img) {

		int YMAX = (int)img.rows;
		int XMAX = (int)img.cols;
		float tan = 0;

		for (int y = 0; y < YMAX; y++) {
			for (int x = 0; x < XMAX; x++) {

				int q = 0;
				int r = 0;
				int edge_x_y = img.at<uchar>(y, x);
				int edge_x_yadd1;
				int edge_xadd1_y;

				if (y + 1 >= YMAX) {
					edge_x_yadd1 = edge_x_y;
				}
				else {
					edge_x_yadd1 = img.at<uchar>(y + 1, x);
				}

				if (x + 1 >= XMAX) {
					edge_xadd1_y = edge_x_y;
				}
				else {
					edge_xadd1_y = img.at<uchar>(y, x + 1);
				}

				if (edge_xadd1_y - edge_x_y != 0) {
					tan = (float)(-(edge_x_yadd1 - edge_x_y)) / (float)(edge_xadd1_y - edge_x_y);
				}
				else {
					tan = 0.0;
				}

				if ((tan >= 0 && tan < 0.4142) || (tan >= -0.4142 && tan <= 0)) {
					if (x + 1 < XMAX) {
						q = img.at<uchar>(y, x + 1);
					}
					if (x - 1 >= 0) {
						r = img.at<uchar>(y, x - 1);
					}
				}
				else if ((tan >= 0.4142 && tan < 2.4142)) {
					if (y - 1 >= 0 && x + 1 < XMAX) {
						q = img.at<uchar>(y - 1, x + 1);
					}
					if (y + 1 < YMAX && x - 1 >= 0) {
						r = img.at<uchar>(y + 1, x - 1);
					}
				}
				else if ((tan >= 2.4142 || tan <= -2.4142)) {
					if (y - 1 >= 0) {
						q = img.at<uchar>(y - 1, x);
					}
					if (y + 1 < YMAX) {
						r = img.at<uchar>(y + 1, x);
					}
				}
				else if ((tan >= -2.4142 && tan < -0.4142)) {
					if (y - 1 >= 0 && x - 1 >= 0) {
						q = img.at<uchar>(y - 1, x - 1);
					}
					if (y + 1 < YMAX && x + 1 < XMAX) {
						r = img.at<uchar>(y + 1, x + 1);
					}
				}

				if (img.at<uchar>(y, x) < q ||
					img.at<uchar>(y, x) < r) {
					img.at<uchar>(y, x) = 0;
				}
			}
		}
    }

	std::vector<cv::Point2f> nodeDetection(cv::UMat src, cv::UMat buffer, int threshold, int maxNodes) {
		cv::boxFilter(src, buffer, -1, cv::Size(3, 3));
		src = buffer;
		cv::threshold(src, buffer, threshold, 255, cv::THRESH_BINARY);
		src = buffer;
		std::vector<cv::Point2f> nodes;
		cv::findNonZero(src, nodes);
		std::vector<cv::Point2f> pickedNodes;
		std::sample(
			nodes.begin(),
			nodes.end(),
			std::back_inserter(pickedNodes),
			maxNodes,
			std::mt19937{ std::random_device{}() }
		);
		return pickedNodes;
	}

	cv::Subdiv2D delaunayTriangulation(std::vector<cv::Point2f> points, cv::Subdiv2D subdiv) {
		subdiv.insert(points);
		return subdiv;
	}

	void drawLines(cv::UMat img, cv::Subdiv2D subdiv) {
		// get triangle list and render
		std::vector<cv::Vec6f> triangleList;
		subdiv.getTriangleList(triangleList);
		std::vector<std::vector<cv::Point>> polygons(
			triangleList.size(), std::vector<cv::Point>(3)
		);
		for (int i = 0; i < polygons.size(); i++) {
			polygons[i][0].x = triangleList[i][0];
			polygons[i][0].y = triangleList[i][1];
			polygons[i][1].x = triangleList[i][2];
			polygons[i][1].y = triangleList[i][3];
			polygons[i][2].x = triangleList[i][4];
			polygons[i][2].y = triangleList[i][5];
		}
		//cv::fillPoly(outputU, polygons, cv::Scalar::all(255));
		cv::polylines(img, polygons, true, cv::Scalar::all(255));
	}
}


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
	lib::opencl::useOpenCL(0, 0, cv::ocl::Device::TYPE_GPU);

	//compile cl code
	cv::ocl::Program nmsProgram = lib::opencl::compile("NonMaxSuppression.cl");

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
		lib::opencl::edgeFilterISD(inputU, outputU);
		imshow("Edge Before Suppressed", outputU);
		inputU = outputU;

        // apply non max suppression
		kNMS.args(
			cv::ocl::KernelArg::ReadWrite(inputU)
		).run(2, globalSize, localSize, true);
		outputU = inputU;   //manually pass input to output because this is a in-place function
		imshow("Edge After Suppressed", outputU);

		// apply node detection
		std::vector<cv::Point2f> pickedNodes = lib::nodeDetection(inputU, outputU, NODE_DETECTION_THRESHOLD, MAX_NODES);

		// apply delaunay triangulation
		cv::Rect rect(0, 0, outputU.size().width, outputU.size().height);
		cv::Subdiv2D subdiv(rect);
		subdiv = lib::delaunayTriangulation(pickedNodes, subdiv);

		// render result of delaunay triangulation
		lib::drawLines(delaunayU, subdiv);

        // show live and wait for a key with timeout long enough to show images
        imshow("delaunay", delaunayU);
        if (cv::waitKey(5) >= 0)
            break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;

}