#include "../include/NodeDetection.h"
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
}