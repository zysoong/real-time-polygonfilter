#include "../include/NonMaxSuppression.h"
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

namespace rp {

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

	void uNonMaxSuppression(cv::ocl::Kernel kernelNMS, size_t* globalSize, 
	size_t* localSize, cv::UMat input, cv::UMat output) {
		kernelNMS.args(
			cv::ocl::KernelArg::ReadWrite(input)
		).run(2, globalSize, localSize, true);
		output = input;   //manually pass input to output because this is a in-place function
	}
	
}