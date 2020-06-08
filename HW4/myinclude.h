
#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <tuple>
#include <opencv2/features2d.hpp>
using namespace std;
using namespace cv;


cv::Mat imageEqualization(void* userdata)
{
	cv::Mat img = *(cv::Mat*)userdata;
	/*
	//Create a window
	namedWindow("window_1", 1);
	cv::imshow("window_1", img);
	*/
	//Converting image in LAB color space
	cvtColor(img, img, COLOR_BGR2Lab);

	// Splitting the image in the LAB channels
	cv::Mat channels[3];
	split(img, channels);

	// Delcaring calcHist parameters
	int histSize = 256; //number of bins

	float range[] = { 0, 256 }; //min and max values of pixel
	const float* histRange = { range };

	bool uniform = true; //bin sizes are the same
	bool accumulate = false; //histogram cleared at the beginning



	cv::Mat channels_out[3];// Variable that will contain the equalized channels of the image

	//EQUALIZATION
	equalizeHist(channels[0], channels_out[0]);//Only Lightness channel is equalized
	channels[1].copyTo(channels_out[1]); //a channel is copied
	channels[2].copyTo(channels_out[2]); //b channel is copied


	cv::Mat img_out;//this variable will contain the final equalized image
	merge(channels_out, 3, img_out);//merge the three equalized channels

	//Converting back to BGR colorspace
	cvtColor(img_out, img_out, COLOR_Lab2BGR);

	//show the final equalized image



	// wait for a key to be pressed and then close all


	return img_out;
}

cv::Mat cylindricalProj(const cv::Mat& image, const double angle)
{
	//cv::Mat tmp, result;
	/*cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
	result = tmp.clone();*/
	cv::Mat channels[3], result[3];
	Mat resultMerged;
	split(image, channels);
	result[0] = channels[0].clone();
	result[1] = channels[1].clone();
	result[2] = channels[2].clone();
	
	double alpha(angle / 180 * CV_PI);
	double d((image.cols / 2.0) / tan(alpha));
	double r(d / cos(alpha));
	double d_by_r(d / r);
	int half_height_image(image.rows / 2);
	int half_width_image(image.cols / 2);

	for (int x = -half_width_image + 1,
		x_end = half_width_image; x < x_end; ++x)
	{
		for (int y = -half_height_image + 1,
			y_end = half_height_image; y < y_end; ++y)
		{
			double x1(d * tan(x / r));
			double y1(y * d_by_r / cos(x / r));

			if (x1 < half_width_image &&
				x1 > -half_width_image + 1 &&
				y1 < half_height_image &&
				y1 > -half_height_image + 1)
			{
				//modified for color compatibility
				result[0].at<uchar>(y + half_height_image, x + half_width_image)
					= channels[0].at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
				result[1].at<uchar>(y + half_height_image, x + half_width_image)
					= channels[1].at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
				result[2].at<uchar>(y + half_height_image, x + half_width_image)
					= channels[2].at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
				merge(result, 3, resultMerged);
			}
		}
	}

	return resultMerged;
}

void maskImages(const Mat& imgleft, const Mat& imgright, Mat& maskRight, Mat& maskLeft) {
	//create mask for search keypoints in only half image
	maskRight = Mat::zeros(imgleft.size(), CV_8U);
	Mat right(maskRight, Rect(imgleft.cols - (imgright.cols*2/3), 0, (imgright.cols * 2/3), imgleft.rows));
	right = Scalar(255);
	maskLeft = Mat::zeros(imgright.size(), CV_8U);
	Mat left(maskLeft, Rect(0, 0, imgright.cols * 2/3, imgright.rows));
	left = Scalar(255);
}

void orbDetection(Mat& img, Mat& mask, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int nfeatures = 300;//300
	int FASTthres = 20;//25
	Ptr<ORB> orb = ORB::create(nfeatures,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,FASTthres);
	//Ptr<AKAZE> akaze = AKAZE::create(); TODO IMPLEMENT AKAZE
	orb->detectAndCompute(img, mask, keypoints, descriptors);
	//akaze->detectAndCompute(img, mask, keypoints, descriptors);
}

void matchingBF(Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches) {
	Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_HAMMING, true));//create matcher
	matcher->match(descriptors1, descriptors2, matches);
}
	
void matchingRefine(vector<DMatch>& matches, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<Point2f>& kp1_refined, vector<Point2f>& kp2_refined ){
	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i < nbMatch; i++)
		tab.at<float>(i, 0) = matches[i].distance;
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;
	int limit = 100;//GOOD
		if (nbMatch < limit) {
			limit = nbMatch;
		}
	for (int i = 0; i < limit; i++)
		bestMatches.push_back(matches[index.at < int >(i, 0)]);
	

	for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
		//-- Get the keypoints from the good matches
		kp1_refined.push_back(keypoints1[it->queryIdx].pt);
		kp2_refined.push_back(keypoints2[it->trainIdx].pt);
	}
	matches = bestMatches;
}

void homogTranslation(Mat& H, vector<Point2f>& kp1_refined, vector<Point2f>& kp2_refined) {
	H = findHomography(kp2_refined, kp1_refined, RANSAC);
	H.at<double>(0, 0) = 1;
	H.at<double>(1, 0) = 0;
	H.at<double>(0, 1) = 0;
	H.at<double>(1, 1) = 1;
	H.at<double>(2, 0) = 0;
	H.at<double>(2, 1) = 0;
	H.at<double>(2, 2) = 1;
}

void alphaBlend(Mat& img1, Mat&img2, Mat& mask, Mat& blended) {
	// Blend img1 and img2 (of CV_8UC3) with mask (CV_8UC1)
	assert(img1.size() == img2.size() && img1.size() == mask.size());
	blended = cv::Mat(img1.size(), img1.type());
	for (int y = 0; y < blended.rows; ++y) {
		for (int x = 0; x < blended.cols; ++x) {
			float alpha = mask.at<unsigned char>(y, x) / 255.0f;
			blended.at<cv::Vec3b>(y, x) = alpha * img1.at<cv::Vec3b>(y, x) + (1 - alpha)*img2.at<cv::Vec3b>(y, x);
		}
	}
}

void connectingImages(Mat& imgleft, Mat& imgright, Mat& H, Mat& result,int firstWidth) {
	Mat rightWarped;
	warpPerspective(imgright, rightWarped, H, cv::Size(firstWidth + H.at<double>(0, 2), imgleft.rows));
	
	//blending mask
	Mat mask(rightWarped.rows, rightWarped.cols, CV_8U, Scalar(255));
	int value = 255;
	for (int c = imgleft.cols-32; c <mask.cols; c++)
	{
		if (value > 0) {
			mask.col(c).setTo(value);
			value = value - 8;
		}
		else
		{
			mask.col(c).setTo(0);
		}
	}


	//resize left part
	Mat biggerLeft(rightWarped.rows, rightWarped.cols, CV_8UC3, Scalar(0));
	Mat leftPart(biggerLeft, Rect(0, 0, imgleft.cols, imgleft.rows));
	imgleft.copyTo(leftPart);
	
	//alpha blending
	alphaBlend(biggerLeft, rightWarped, mask, result);
}

void addRight(Mat& imgleft, Mat& imgright, Mat& result,int firstWidth,int i){
	//create mask for search keypoints in only half image
	Mat maskRight, maskLeft;
	std::cout << "creating Masks"<<endl;
	maskImages(imgleft, imgright, maskRight, maskLeft);

	//cylindrical projection
	std::cout << "performing cylindrical projection" << endl;
	imgright = cylindricalProj(imgright, 33);


	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	std::cout << "running ORB detection"<<endl;
	orbDetection(imgleft, maskRight, keypoints1, descriptors1);
	orbDetection(imgright, maskLeft, keypoints2, descriptors2);

	vector<DMatch> matches;
	std::cout << "computing matches" << endl;
	matchingBF(descriptors1, descriptors2, matches);
	std::vector<Point2f> kp1_refined;
	std::vector<Point2f> kp2_refined;
	std::cout << "matching refinement" << endl;
	matchingRefine(matches, keypoints1, keypoints2, kp1_refined, kp2_refined);
	
	//Drawing matches
	Mat comparison;
	cv::drawMatches(imgleft, keypoints1, imgright, keypoints2, matches, comparison);
	string filename = "comparison " + to_string(i-1)+"-"+ to_string(i)+".png";
	imwrite(filename, comparison);
	//cv::imshow(filename, comparison);
	//cv::waitKey();

	//extract translation matrix from homography
	Mat H;
	std::cout << "estimating translation"<<endl;
	homogTranslation(H, kp1_refined, kp2_refined);
	std::cout << H << endl;
	std::cout << "connecting images"<<endl;
	connectingImages(imgleft, imgright, H, result,firstWidth);
	/*imshow("result", result);
	waitKey();
	destroyWindow(filename);*/
}