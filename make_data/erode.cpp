#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat element3 = getStructuringElement(MORPH_RECT, Size(7, 7));
Mat element4 = getStructuringElement(MORPH_CROSS, Size(9, 9));

int main(){
	Mat src = imread("./train_data3/01.png");

	Mat out1;
	dilate(src, out1, element1);
	imwrite("./train_data3/02.png", out1);
	Mat out2;
	dilate(src, out2, element2);
	imwrite("./train_data3/03.png", out2);
	Mat out3;
	dilate(src, out3, element3);
	imwrite("./train_data3/04.png", out3);
	Mat out4;
	dilate(src, out4, element4);
	imwrite("./train_data3/05.png", out4);

	Mat out5;
	erode(src, out5, element1);
	imwrite("./train_data3/06.png", out5);
	Mat out6;
	erode(src, out6, element2);
	imwrite("./train_data3/07.png", out6);
	Mat out7;
	erode(src, out7, element3);
	imwrite("./train_data3/08.png", out7);
	Mat out8;
	erode(src, out8, element4);
	imwrite("./train_data3/09.png", out8);
	
//	waitKey(0);
	return 0;
}
