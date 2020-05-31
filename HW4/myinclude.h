
#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

Mat imageEqualization(void* userdata);

Mat cylindricalProj(const cv::Mat& image, const double angle);
