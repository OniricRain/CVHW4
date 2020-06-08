#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include "myinclude.h"

using namespace std;
using namespace cv;


// starting point for the application
int main(int argc, char** argv)//TODO control this
{
	vector<cv::String> filepath;
	// load an image and place it in the img variable
	//glob("lab/*.bmp",filepath, false);
	glob("kitchen/*.bmp", filepath, false);

	Mat imgleft = cv::imread(filepath[6]);
	int firstWidth = imgleft.cols;
	imgleft = cylindricalProj(imgleft, 33);
	//imshow("testcolor", imgleft);
	imgleft = imageEqualization((void*)&imgleft);


	//TODO try CLAHE
	int n = filepath.size();
	//int n = 9;
	Mat result;
	for (size_t i = 7; i < n; i++)
	{
		std::cout << "Processing image number " << i << endl;
		Mat imgright = cv::imread(filepath[i]);
		imgright = imageEqualization((void*)&imgright);
		addRight(imgleft, imgright, result,firstWidth,i);
		imgleft = result;
	}
	//cv::imshow("result", result);
	imwrite("resultBlended.png", result);
	

	/*
	//-- Draw keypoints
	Mat img_keypoints;
	drawKeypoints(res, keypoints1, img_keypoints);
	//-- Show detected (drawn) keypoints
	imshow("ORB Keypoints", img_keypoints);
	*/

	/*
	Mat img_out = imageEqualization((void*)&img);
	//show the final equalized image
	namedWindow("window_2", 1);
	cv::imshow("window_2", img_out);*/
	

	// wait for a key to be pressed and then close all
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}