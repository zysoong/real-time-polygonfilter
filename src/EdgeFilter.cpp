#include "../include/EdgeFilter.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace rp {
	void edgeFilterISD(cv::UMat src, cv::UMat dst) {
		cv::Mat kernelMat = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
		cv::UMat kernelUMat = kernelMat.getUMat(cv::ACCESS_READ);
		cv::filter2D(src, dst, -1, kernelUMat);
	}
}