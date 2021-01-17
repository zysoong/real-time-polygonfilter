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
	std::vector<cv::Point2f> nodeDetection(cv::UMat src, cv::UMat buffer, int threshold, int maxNodes);
}