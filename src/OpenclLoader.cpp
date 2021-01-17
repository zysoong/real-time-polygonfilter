#include "../include/OpenclLoader.h"
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <fstream>
#include <direct.h>
#include <stdlib.h>
#include <errno.h>
#include <filesystem>

namespace rp {

	namespace fs = std::filesystem;

	void printPlatformNames() {
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
		if (!errmsg.empty()) {
			std::cout << "OpenCL program build log:" << std::endl << errmsg << std::endl;
		}
		return program;
	}
}