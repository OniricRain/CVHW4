#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;



Mat imageEqualization(void* userdata)
{
	Mat img = *(Mat*)userdata;
	//Create a window
	namedWindow("window_1", 1);
	cv::imshow("window_1", img);

	//Converting image in LAB color space
	cvtColor(img, img, COLOR_BGR2Lab);

	// Splitting the image in the LAB channels
	Mat channels[3];
	split(img, channels);

	// Delcaring calcHist parameters
	int histSize = 256; //number of bins

	float range[] = { 0, 256 }; //min and max values of pixel
	const float* histRange = { range };

	bool uniform = true; //bin sizes are the same
	bool accumulate = false; //histogram cleared at the beginning



	Mat channels_out[3];// Variable that will contain the equalized channels of the image

	//EQUALIZATION
	equalizeHist(channels[0], channels_out[0]);//Only Lightness channel is equalized
	channels[1].copyTo(channels_out[1]); //a channel is copied
	channels[2].copyTo(channels_out[2]); //b channel is copied


	Mat img_out;//this variable will contain the final equalized image
	merge(channels_out, 3, img_out);//merge the three equalized channels

	//Converting back to BGR colorspace
	cvtColor(img_out, img_out, COLOR_Lab2BGR);

	//show the final equalized image



	// wait for a key to be pressed and then close all


	return img_out;
}