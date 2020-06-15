
#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <tuple>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;
using namespace cv;

/**
 * @brief  Selection between two descriptors
 * 
 * @return int the selected descriptor
 */
void selectDescSet(int& desc, int& set) {
	
	do {
		cout << "select descriptor, type a number:" << endl;
		cout << "1) ORB" << endl << "2) AKAZE" << endl;
		cin >> desc;
		//desc = 1; //TODO remove
	} while (desc != 1 && desc != 2);
	do {
		cout << "select imageset, type a number:" << endl;
		cout << "1) Lab" << endl << "2) Kitchen" << endl << "3) Dolomites" << endl << "4) Fountain" << endl;
		cin >> set;
		//set = 4;//TODO remmove
	} while (set != 1 && set != 2 && set != 3 && set != 4) ;
}

/**
 * @brief performs histogram equalization on an input image,
 * the equalization is done in the LAB colorspace, only on the L channel
 * 
 * @param userdata the input image
 * @return cv::Mat the equalized image
 */
cv::Mat imageEqualization(void* userdata)
{
	cv::Mat img = *(cv::Mat*)userdata;

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

	return img_out;
}

/**
 * @brief Perform a cylindrical projection on a given input image
 * the projection is done on each RGB channel, then the channels are merged
 * 
 * @param image input image
 * @param angle is the camera Field of View divided by 2
 * @return cv::Mat output image
 */
cv::Mat cylindricalProj(const cv::Mat& image, const double angle)
{
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
				//modified for color compatibility, perform the projection on each channel
				result[0].at<uchar>(y + half_height_image, x + half_width_image)
					= channels[0].at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
				result[1].at<uchar>(y + half_height_image, x + half_width_image)
					= channels[1].at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
				result[2].at<uchar>(y + half_height_image, x + half_width_image)
					= channels[2].at<uchar>(round(y1 + half_height_image),
						round(x1 + half_width_image));
				merge(result, 3, resultMerged); // merging the three projected channels
			}
		}
	}

	return resultMerged;
}

/**
 * @brief program inizialization, selection of the imageset and
 * preprocessing (cylindrical projection and histogram equalization)
 * of the first image
 * 
 * @param filepath path of the selected imageset
 * @param imgleft first image of the panoramic
 */
void acquisition(vector<cv::String>& filepath, Mat& imgleft, int& set) {
	switch (set)
	{
		case 1://lab set
		{
			glob("lab/*.bmp", filepath, false);
			//glob("pL/*.bmp", filepath, false);
			break;
		}
		case 2://kitchen set
		{
			//glob("pK/*.bmp", filepath, false);
			glob("kitchen/*.bmp", filepath, false);
			break;
		}
		case 3://dolomites set
		{
			glob("dolomites/*.png", filepath, false);
			break;
		}
		case 4://fountain set
		{
			glob("fountain/*.jpg", filepath, false);
			//glob("pM/*.jpg", filepath, false);
			break;
		}
	default:
		break;
	}



	imgleft = cv::imread(filepath[0]);
	imgleft = cylindricalProj(imgleft, 33);// TODO UNCOMMENT//51/2 sony
	imgleft = imageEqualization((void*)&imgleft);// TODO UNCOMMENT
	
	//string path = "pM/00.jpg";
	//imwrite(path, imgleft);
}

/**
 * @brief creating a mask which allows to select only 2/3 of the image
 * the descriptors will run only in this region of interest for each image
 * left image mask -> 2/3 on the right side
 * right image mask -> 2/3 on the left side
 *  
 * @param imgleft left image
 * @param imgright right image
 * @param maskRight ROI of left image
 * @param maskLeft ROI of right image
 */
void maskImages(const Mat& imgleft, const Mat& imgright, Mat& maskRight, Mat& maskLeft) {
	//create mask for search keypoints in only half image
	maskRight = Mat::zeros(imgleft.size(), CV_8U);
	Mat right(maskRight, Rect(imgleft.cols - (imgright.cols*2/3), 0, (imgright.cols * 2/3), imgleft.rows));
	right = Scalar(255);
	maskLeft = Mat::zeros(imgright.size(), CV_8U);
	Mat left(maskLeft, Rect(0, 0, imgright.cols * 2/3, imgright.rows));
	left = Scalar(255);
	imwrite("right.png", maskLeft);
	imwrite("left.png", maskRight);
}

/**
 * @brief initialize the ORB descriptor with parameters and perform feature extraction and descritption
 * 
 * @param img input image
 * @param mask relative mask computed with maskImages
 * @param keypoints output keypoints
 * @param descriptors output descriptors
 * @param num_matches for testing purposes, set the number of features to detect
 */
void orbDetComp(Mat& img, Mat& mask, vector<KeyPoint>& keypoints, Mat& descriptors,int num_matches, int& set,int i) {
	int nfeatures;
	switch (set)
	{
	case 1://lab set
	{
		nfeatures = 340;//300
		break;
	}
	case 2://kitchen set
	{
		nfeatures = 500;
		break;
	}
	case 3://dolomites set
	{
		nfeatures = 500;
		break;
	}
	case 4://fountain set
	{
		nfeatures = 410;
		break;
	}
	default:
		break;
	}
	int FASTthres = 20;
	Ptr<ORB> orb = ORB::create(nfeatures,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,FASTthres);
	orb->detectAndCompute(img, mask, keypoints, descriptors);
}


/**
 * @brief initialize the AKAZE descriptor with paramers and perform feature extraction and descritption
 * 
 * @param img input image
 * @param mask relative mask computed with maskImages
 * @param keypoints output keypoints
 * @param descriptors output descriptors
 * @param num_matches //TODO remove
 */
void akazeDetComp(Mat& img, Mat& mask, vector<KeyPoint>& keypoints, Mat& descriptors, int num_matches, int& set) {
	float thres;
	switch (set)
	{
	case 1://lab set
	{
		thres = 1/float(480);
		break;
	}
	case 2://kitchen set
	{
		thres = 1 / float(320);
		break;
	}	 
	case 3://dolomites set
	{	 
		thres = 0.001;
		break;
	}
	case 4://fountain set
	{
		thres = 1 / float(460);
		break;
	}
	default:
		break;
	}
	Ptr<AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB,0,3,thres,4,4,KAZE::DIFF_PM_G2);
	akaze->detectAndCompute(img, mask, keypoints, descriptors);
}

/**
 * @brief Brute-Force feature matching
 * 
 * @param descriptors1 descriptors computed for left image
 * @param descriptors2 descriptors computed for right image
 * @param matches output vector of matches
 */
void matchingBF(Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches) {
	Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_HAMMING, true));//create matcher
	matcher->match(descriptors1, descriptors2, matches);
}

/**
 * @brief experimental procedure to filter out the matches with minimum hamming distance,
 * the objective was to use RANSAC only on these matches.
 * Since this procedure didn't improve results, this function is no longer used
 * 
 * @param matches matches founded by matchingBF
 * @return vector<DMatch> output matches with minimum distance
 */
vector<DMatch> minHamming(vector<DMatch>& matches) {
	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i < nbMatch; i++)
		tab.at<float>(i, 0) = matches[i].distance;
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;
	int limit = 100;
	if (nbMatch < limit) {
		limit = nbMatch;
	}
	for (int i = 0; i < limit; i++)
		bestMatches.push_back(matches[index.at < int >(i, 0)]);
	return bestMatches;
}

/**
 * @brief filter the keypoints that have been matched
 * 
 * @param matches matches founded by matchingBF
 * @param keypoints1 keypoints computed for left image
 * @param keypoints2 keypoints computed for right image
 * @param kp1_refined output keypoints for left image 
 * @param kp2_refined output keypoints for right image
 * @param num_matches //TODO remove
 */
void matchingRefine(vector<DMatch>& matches, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<Point2f>& kp1_refined, vector<Point2f>& kp2_refined, int num_matches ){
	//vector<DMatch> bestMatches = minHamming(matches); // no longer used

	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); ++it) {
		//-- Get the keypoints from the good matches
		kp1_refined.push_back(keypoints1[it->queryIdx].pt);
		kp2_refined.push_back(keypoints2[it->trainIdx].pt);
	}
}

/**
 * @brief compute homography and select only the transaltion parameters
 * an homography matrix which performs only translation looks like:
 * [1 0 tx]
 * [0 1 ty]
 * [0 0 1]
 * @param H output homography matrix
 * @param kp1_refined keypoints for left image
 * @param kp2_refined keypoints for right image
 */
void homogTranslation(Mat& H, vector<Point2f>& kp1_refined, vector<Point2f>& kp2_refined) {
	H = findHomography(kp2_refined, kp1_refined, RANSAC,3);
	cout << H << endl;
	H.at<double>(0, 0) = 1;
	H.at<double>(1, 0) = 0;
	H.at<double>(0, 1) = 0;
	H.at<double>(1, 1) = 1;
	H.at<double>(2, 0) = 0;
	H.at<double>(2, 1) = 0;
	H.at<double>(2, 2) = 1;
}

/**
 * @brief perform the alpha blend technique in order to have a gradual transaction
 * from the left image to the right one.
 * 
 * @param img1 left image
 * @param img2 right image
 * @param mask mask with parameter for the alphablend transaction
 * @param blended resulted blend image
 */
void alphaBlend(Mat& img1, Mat&img2, Mat& mask, Mat& blended) {
	// Blend img1 and img2 (of CV_8UC3) with mask (CV_8UC1)
	assert(img1.size() == img2.size() && img1.size() == mask.size()); //sizes of img1, img2 and mask must coincide
	blended = cv::Mat(img1.size(), img1.type());
	for (int y = 0; y < blended.rows; ++y) {
		for (int x = 0; x < blended.cols; ++x) {
			float alpha = mask.at<unsigned char>(y, x) / 255.0f;
			blended.at<cv::Vec3b>(y, x) = alpha * img1.at<cv::Vec3b>(y, x) + (1 - alpha)*img2.at<cv::Vec3b>(y, x);
		}
	}
}

/**
 * @brief after computing the warping of the right image with respect to the left (using translation matrix)
 * connects the panoramic image with the new right part, using the alphaBlend class
 * 
 * @param imgleft left image
 * @param imgright right image
 * @param H transaltion matrix
 * @param result old panoramic image in input, new panoramic image in output
 */
void connectingImages(Mat& imgleft, Mat& imgright, Mat& H, Mat& result) {
	Mat rightWarped;
	warpPerspective(imgright, rightWarped, H, cv::Size(imgleft.cols + H.at<double>(0, 2), imgleft.rows));
	imwrite("rightwarped.png", rightWarped);
	//resize right part
	Mat biggerRight(result.rows, result.cols + H.at<double>(0, 2), CV_8UC3, Scalar(0));
	Mat rightPart(biggerRight, Rect(biggerRight.cols - rightWarped.cols, 0, rightWarped.cols, rightWarped.rows));
	rightWarped.copyTo(rightPart);
	
	//resize left part
	Mat biggerLeft(result.rows, result.cols + H.at<double>(0, 2), CV_8UC3, Scalar(0));
	Mat leftPart(biggerLeft, Rect(0, 0, result.cols, result.rows));
	result.copyTo(leftPart);
	
	//blending mask
	Mat mask(result.rows, result.cols + H.at<double>(0, 2), CV_8U, Scalar(255));
	int value = 255;
	for (int c = result.cols-32; c <mask.cols; c++)
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
	//imwrite("mask.png", mask);
	//alpha blending
	alphaBlend(biggerLeft, biggerRight, mask, result);
}

/**
 * @brief perform an iteration of the panoramic stitching process, adding an image to the right
 * 
 * @param imgleft the last image addedd to the panorama
 * @param imgright the new image to add
 * @param result the panorama so far, in the output will be the new panorama
 * @param i number of the iteration, and the number of the image to add
 * @param num_matches for testing purposes, set the number of features to detect in ORB
 * @param yshift keep track of the shift on the y axis, will use for cut out the black borders
 * @param descriptor used descriptor, ORB or AKAZE
 */
void addRight(Mat& imgleft, Mat& imgright, Mat& result,int i,int num_matches, double& yshift, int& descriptor, int& set){
	//create mask for search keypoints in only half image
	Mat maskRight, maskLeft;
	std::cout << "creating Masks"<<endl;
	maskImages(imgleft, imgright, maskRight, maskLeft);


	// TODO UNCOMMENT
	
	//cylindrical projection and equalization
	std::cout << "performing cylindrical projection" << endl;
	imgright = cylindricalProj(imgright, 33);
	imgright = imageEqualization((void*)&imgright);
	/*
	string path = "pM/0"+to_string(i)+".jpg";
	imwrite(path, imgright);*/

	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	if (descriptor == 1) {
		std::cout << "running ORB detection" << endl;

		orbDetComp(imgleft, maskRight, keypoints1, descriptors1, num_matches,set,i);
		orbDetComp(imgright, maskLeft, keypoints2, descriptors2, num_matches,set,i);
	}
	else if (descriptor == 2)
	{
		std::cout << "running AKAZE detection and computing" << endl;
		akazeDetComp(imgleft, maskRight, keypoints1, descriptors1, num_matches, set);
		akazeDetComp(imgright, maskLeft, keypoints2, descriptors2, num_matches, set);
	}
	vector<DMatch> matches;
	std::cout << "computing matches" << endl;
	matchingBF(descriptors1, descriptors2, matches);
	std::vector<Point2f> kp1_refined;
	std::vector<Point2f> kp2_refined;
	std::cout << "matching refinement" << endl;
	matchingRefine(matches, keypoints1, keypoints2, kp1_refined, kp2_refined, num_matches);
	
	//Drawing matches
	Mat comparison;
	cv::drawMatches(imgleft, keypoints1, imgright, keypoints2, matches, comparison);
	string filename = "comparison " + to_string(i-1)+"-"+ to_string(i)+".png";
	imwrite(filename, comparison);
	cv::imshow("comparison", comparison);
	cv::waitKey(1);

	//extract translation matrix from homography
	Mat H;
	std::cout << "estimating translation"<<endl;
	homogTranslation(H, kp1_refined, kp2_refined);
	//cout << "estimating homography"<<endl;
	//H = findHomography(kp2_refined, kp1_refined, RANSAC, 3);
	std::cout << H << endl;
	std::cout << "connecting images"<<endl;
	connectingImages(imgleft, imgright, H, result);
	yshift = H.at<double>(1, 2);

	
	//cv::imshow("result", result);
	//cv::waitKey();
	//cv::destroyWindow(filename);
	
}

/**
 * @brief update the shift on the y axis during this iteration of addRight class
 * 
 * @param yshift the shift in this iteration
 * @param oshiftPlus the maximum positive shift obtained in this imageset
 * @param oshiftMinus the minimum negative shift obtained in this imageset
 */
void updateShift(double yshift, double& oshiftPlus, double& oshiftMinus) {
	if (yshift > oshiftPlus) {
		oshiftPlus = yshift;
	}
	else if (yshift < oshiftMinus)
	{
		oshiftMinus = yshift;
	}
}

/**
 * @brief cut out the black borders obtained due the shift on the y axis
 * the final result will be writed as "cut.png"
 * 
 * @param result input image with black borders
 * @param oshiftPlus the maximum positive shift obtained in this imageset
 * @param oshiftMinus  the minimum negative shift obtained in this imageset
 */
void lastCut(Mat& result, double oshiftPlus, double oshiftMinus) {
	cout << "Positive shift " << oshiftPlus << endl;
	cout << "Negative shift" << oshiftMinus << endl;
	//cv::imshow("result", result);
	imwrite("resultBlended.png", result);
	Mat cut(result, Rect(0, oshiftPlus, result.cols, result.rows + oshiftMinus - oshiftPlus));
	imwrite("cut.png", cut);
}

