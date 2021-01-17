#include "../include/DelaunayTriangulation.h"
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