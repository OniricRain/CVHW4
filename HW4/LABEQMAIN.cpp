#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;


void showHistogram(std::vector<cv::Mat>& hists);

// starting point for the application
int main(int argc, char** argv)//TODO control this
{
	// load an image and place it in the img variable
	cv::Mat img = cv::imread("barbecue.png");


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

	/*
	Mat l_hist, a_hist, b_hist; //output matrixes

	//Calculate histogram for each channel (LAB)
	calcHist(&channels[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	//save the histograms in a vector of 3 cv::mat
	vector<Mat> hists;
	hists.push_back(l_hist);
	hists.push_back(a_hist);
	hists.push_back(b_hist);

	//Show the three histograms
	showHistogram(hists);*/

	Mat channels_out[3];// Variable that will contain the equalized channels of the image

	//EQUALIZATION
	equalizeHist(channels[0], channels_out[0]);//Only Lightness channel is equalized
	channels[1].copyTo(channels_out[1]); //a channel is copied
	channels[2].copyTo(channels_out[2]); //b channel is copied

	/*
	//caluclate histogram for each channel after equalization
	calcHist(&channels_out[0], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels_out[1], 1, 0, Mat(), a_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&channels_out[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	vector<Mat> hists_equalized;//save the equalized histograms in a vector of 3 cv::mat
	hists_equalized.push_back(l_hist);
	hists_equalized.push_back(a_hist);
	hists_equalized.push_back(b_hist);

	//Show the three equalized histograms
	showHistogram(hists_equalized);*/

	Mat img_out;//this variable will contain the final equalized image
	merge(channels_out, 3, img_out);//merge the three equalized channels

	//Converting back to BGR colorspace
	cvtColor(img_out, img_out, COLOR_Lab2BGR);

	//show the final equalized image
	namedWindow("window_2", 1);
	cv::imshow("window_2", img_out);
	imwrite("barbecueout.png", img_out);

	// wait for a key to be pressed and then close all
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
	// Min/Max computation
	double hmax[3] = { 0,0,0 };
	double min;
	cv::minMaxLoc(hists[0], &min, &hmax[0]);
	cv::minMaxLoc(hists[1], &min, &hmax[1]);
	cv::minMaxLoc(hists[2], &min, &hmax[2]);

	std::string wname[3] = { "Lightness", "A", "B" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
							 cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());

	// Display each histogram in a canvas
	for (int i = 0, end = hists.size(); i < end; i++)
	{
		canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
				hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
	}

	//imwrite("histr.jpg", canvas[0]);
	//imwrite("histg.jpg", canvas[1]);
	//imwrite("histb.jpg", canvas[2]);
}
