#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void preprocess(Mat &src);

int main(){
	//Mat src = imread("/mnt/hgfs/USHARE/test_data3_origin/e2.jpg", 0);
	Mat src = imread("./train_data/51.png", 0);
	src = src < 100;
	float x_cdt = 40, y_cdt = 40, rate = 10;
	float angle = 70 * CV_PI / 180;
	float angle2 = 20 * CV_PI / 180;
	Point2f psrc[3] = {Point2f(x_cdt, y_cdt), Point2f(x_cdt, y_cdt-rate), 
		               Point2f(x_cdt+rate, y_cdt)};
    Point2f pdst[3] = {Point2f(x_cdt, y_cdt), 
		               Point2f(x_cdt-rate*cos(angle),y_cdt-rate*sin(angle)),
		               Point2f(x_cdt+rate*cos(angle2),y_cdt-rate*sin(angle2))};
	Mat rect = getAffineTransform(psrc, pdst);
	Mat out;
	warpAffine(src, out, rect, src.size());		
	imshow("sss", out);
	imwrite("./test_data3/51.png", out);
	waitKey(0);
	return 0;
}

