#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <fstream>
#include <direct.h>
#include <stdlib.h>
#include <errno.h>
#include <filesystem>

namespace rp {
	void printPlatformNames();
	cv::ocl::Context useOpenCL(int platformIndex, int deviceIndex, int deviceType);
	cv::ocl::Program compile(std::string clFile);
}