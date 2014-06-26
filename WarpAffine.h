#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <windows.h>
//#include "utils/utils.h"
using namespace std;
using namespace cv;
void DrawLabelsMask(Mat& imgLabel,vector<Point2d>& points,vector<vector<size_t>>& triangles);
void WarpAffine(Mat& img,vector<Point2d>& s_0,vector<Point2d>& s_1, vector<vector<size_t>>& triangles, Mat& dstLabelsMask,Mat& dst);
void CalcCoeffs(vector<Point2d>& s_0,vector<Point2d>& s_1, vector<vector<size_t>>& triangles, Mat& Coeffs);
cv::Rect_<double> boundingRect(vector<Point2d>& pts);