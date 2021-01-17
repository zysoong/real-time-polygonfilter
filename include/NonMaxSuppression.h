#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/opengl.hpp>

namespace rp {
	void nonMaxSuppresion(cv::Mat img);
	void uNonMaxSuppression(cv::ocl::Kernel kernelNMS, size_t* globalSize, 
		size_t* localSize, cv::UMat input, cv::UMat output);
}