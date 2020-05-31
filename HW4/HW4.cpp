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
	// load an image and place it in the img variable
	cv::Mat img1 = cv::imread("lab/i01.bmp");
	cv::Mat img2 = cv::imread("lab/i02.bmp");
	//create mask for search keypoints in only half image
	Mat maskRight = Mat::zeros(img1.size(), CV_8U);
	Mat right(maskRight, Rect(img1.cols / 2, 0, img1.cols / 2, img1.rows));
	right = Scalar(255);
	Mat maskLeft = Mat::zeros(img2.size(), CV_8U);
	Mat left(maskLeft, Rect(0, 0, img2.cols / 2, img2.rows));
	left = Scalar(255);

	Mat res1, res2;
	res1 = cylindricalProj(img1, 33);
	res2 = cylindricalProj(img2, 33);
	//cv::imshow("1", res1);
	//cv::imshow("2", res2);

	Ptr<ORB> orb = ORB::create();
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	orb->detectAndCompute(res1, maskRight, keypoints1, descriptors1);
	orb->detectAndCompute(res2, maskLeft, keypoints2, descriptors2);

	Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_HAMMING, true));//create matcher
	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);
	Mat comparison;
	//draw matches
	drawMatches(res1, keypoints1, res2, keypoints2, matches, comparison);
	imshow("3", comparison);

	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i < nbMatch; i++)
		tab.at<float>(i, 0) = matches[i].distance;
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;

	for (int i = 0; i < 150; i++)
		bestMatches.push_back(matches[index.at < int >(i, 0)]);


	// 1st image is the destination image and the 2nd image is the src image
	std::vector<Point2f> dst_pts;                   //1st
	std::vector<Point2f> source_pts;                //2nd

	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
		//-- Get the keypoints from the good matches
		dst_pts.push_back(keypoints1[it->queryIdx].pt);
		source_pts.push_back(keypoints2[it->trainIdx].pt);
	}



	Mat H = findHomography(source_pts, dst_pts, RANSAC);
	cv::Mat result;
	warpPerspective(res2, result, H, cv::Size(res1.cols + res2.cols, res1.rows));
	cv::Mat half(result, cv::Rect(0, 0, res2.cols, res2.rows));
	res1.copyTo(half);
	imshow("Result", result);


	//TODO: remove black borders.


	/*
	//-- Draw keypoints
	Mat img_keypoints;
	drawKeypoints(res, keypoints1, img_keypoints);
	//-- Show detected (drawn) keypoints
	imshow("ORB Keypoints", img_keypoints);
	*/

	/*Mat img_out = imageEqualization((void*)&img);

	//show the final equalized image
	namedWindow("window_2", 1);
	cv::imshow("window_2", img_out);
	imwrite("barbecueout.png", img_out);*/

	// wait for a key to be pressed and then close all
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}