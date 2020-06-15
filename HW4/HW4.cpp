#include <iostream>

//include openCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include "myinclude.h"

using namespace std;
using namespace cv;


// starting point for the application
int main()
{
	int desc, set;
	selectDescSet(desc,set);
	
	vector<cv::String> filepath;
	Mat imgleft;
	vector<double> shifts;
	double yshift, oshiftPlus,oshiftMinus;
	for (size_t num_matches =800; num_matches <= 800; num_matches += 10)
	{
		oshiftPlus = 0;
		oshiftMinus = 0;
		acquisition(filepath, imgleft,set);
			   		
		int n = filepath.size();
		//int n = 14;
		Mat result = imgleft;
		for (size_t i = 1; i < n; i++)
		{
			yshift = 0;
			std::cout << "Processing image number " << i << " NUM MATCHES " << num_matches << endl;
			Mat imgright = cv::imread(filepath[i]);
			addRight(imgleft, imgright, result, i, num_matches, yshift, desc, set);
			updateShift(yshift, oshiftPlus, oshiftMinus);
			imgleft = imgright;
		}
		shifts.push_back(std::max(std::abs(oshiftMinus), oshiftPlus));
		lastCut(result, oshiftPlus, oshiftMinus);// TODO uncomment
	}


	

	// wait for a key to be pressed and then close all
	cout << "all the shifts:" << endl;
	int num_matches = 150;
	for (std::vector<double>::const_iterator i = shifts.begin(); i != shifts.end(); ++i) {
		std::cout << "num " << num_matches << " " << *i << endl;
		num_matches += 10;
	}
	

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}