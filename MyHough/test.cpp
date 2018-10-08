#include "MyHough.h"

int main(){

	Mat test = imread("../test_digit_origin/43.jpg", 0);	 
	
//	imshow("test", test);
	test.convertTo(test, CV_32F);
	int piece = 5;
	MyHough hough = MyHough(test, piece);
	float ag1 = 0, ag2 = 0;
	hough.get_direction(ag1, ag2);

	waitKey(0);
	return 0;
}

