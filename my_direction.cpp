#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

void x_gradient(const Mat &src, Mat &dst);
void y_gradient(const Mat &src, Mat &dst);
void gradient(const Mat &x_src, const Mat &y_src, Mat &g_dst, Mat &l_dst);
void my_init(Mat &src, Mat &src2, Mat &src3, Mat &src4);
void get_direction(const Mat &src, float &dir1, float &dir2);

void hof(const Mat &src);

int main(){
	//Mat test = imread("./test_data/33.jpg", 0);
	//Mat test = imread("./test_data/21.jpg", 0);
	//Mat test = imread("./test_data/61.jpg", 0);
	Mat test = imread("./test_data/41.jpg", 0);
	
	imshow("origin", test);
	
	hof(test);

	cout << "---------------------" << endl;

	test.convertTo(test, CV_32F);
	//cout << test << endl;	
	
	Mat xg = test.clone();
	Mat yg = test.clone();
	Mat gg = test.clone();
	Mat ll = test.clone();
	my_init(xg, yg, gg, ll);
	x_gradient(test, xg);
	y_gradient(test, yg);
	gradient(xg, yg, gg, ll);	
	cout << gg << endl;

	float angle = 0;
	float angle2 = 0;
	get_direction(gg, angle, angle2);
	cout << "angle: " << angle << endl;
	cout << "angle2: " << angle2 << endl;

	waitKey(0);
	return 0;
}

void get_direction(const Mat &src, float &d, float &dd){
	float dir1[3] = {};
	float dir2[3] = {};
	int num1[3] = {};
	int num2[3] = {};
	for(int i = 0; i < src.rows; ++i){
	 	for(int j = 0; j < src.cols; ++j){
			float temp = src.at<float>(i, j);
			//right
			if((temp > 0) && (temp <= 10)){
				num1[0]++;
				//cout << temp << endl;
				dir1[0] += temp;
			} else if((temp > 10) && (temp <= 20)){
				num1[1]++;
				dir1[1] += temp;
			} else if((temp > 20) && (temp <= 30)){
				num1[2]++;
				dir1[2] += temp;
			} else if((temp > 90) && (temp <= 100)){
				num2[0]++;
				dir2[0] += temp;
			} else if((temp > 100) && (temp <= 110)){
				num2[1]++;
				dir2[1] += temp;
			} else if((temp > 110) && (temp <= 120)){
				num2[2]++;
				dir2[2] += temp;
			} else {
				continue;
			}
		}

	}
	for(int i = 0; i < 3; ++i){
		if(num1[i] != 0){
			dir1[i] = dir1[i] / num1[i];
		}
		if(num2[i] != 0){
			dir2[i] = dir2[i] / num2[i];
		}
	}
	float temp1 = 0, temp2 = 0;
	for(int i = 0; i < 3; ++i){
		temp1 += num1[i];
		temp2 += num2[i];
	}
	for(int i = 0; i < 3; ++i){
		d += dir1[i] * (num1[i] / temp1);
		dd += dir2[i] * (num2[i] / temp2);
	}
	cout << "num1: " << num1[0] << " dir1: " << dir1[0] << endl;
	cout << "num1: " << num1[1] << " dir1: " << dir1[1] << endl;
	cout << "num1: " << num1[2] << " dir1: " << dir1[2] << endl;
	cout << "num2: " << num2[0] << " dir2: " << dir2[0] << endl;
	cout << "num2: " << num2[1] << " dir2: " << dir2[1] << endl;
	cout << "num2: " << num2[2] << " dir2: " << dir2[2] << endl;
	cout << "find direction OK" << endl;	
}

void gradient(const Mat &x_src, const Mat &y_src, Mat &dst, Mat &dst2){
	for(int i = 0; i < x_src.rows; ++i){
	 	for(int j = 0; j < x_src.cols; ++j){
			float a = x_src.at<float>(i, j);
			float b = y_src.at<float>(i, j);
			dst.at<float>(i, j) = atan2(b, a) * 180 / CV_PI;
			dst2.at<float>(i, j) = pow((pow(a, 2) + pow(b, 2)), 0.5);
		}
	}
}

void x_gradient(const Mat &src, Mat &dst){
	int m = src.rows;
	int n = src.cols;
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			if(j == n-1){
				dst.at<float>(i, j) = 0;
			} else {
				dst.at<float>(i, j) = (src.at<float>(i,j) - 
							           src.at<float>(i, j+1)) * 
						              src.at<float>(i, j);
			}
		}
	}
}

void y_gradient(const Mat &src, Mat &dst){
	int m = src.rows;
	int n = src.cols;
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			if(i == m-1){
				dst.at<float>(i, j) = 0;
			} else {
				dst.at<float>(i, j) = (src.at<float>(i, j) - 
			                           src.at<float>(i+1, j)) * 
				                      src.at<float>(i, j);
			}
		}
	}
}

void my_init(Mat &src, Mat &src2, Mat &src3, Mat &src4){
	for(int i = 0; i < src.rows; ++i){
	 	for(int j = 0; j < src.cols; ++j){
			src.at<float>(i, j) = 0;
			src2.at<float>(i, j) = 0;
			src3.at<float>(i, j) = 0;
			src4.at<float>(i, j) = 0;
		}
	}
}

void hof(const  Mat &src){
	Mat canny, out, rect;
	vector<Vec2f> lines;
	Canny(src, canny, 3, 9);
	HoughLines(canny, lines, 1, CV_PI/180, 25);
	float angle = 0, angle2 = 0;
	int num = 0, num2 = 0;
	for(size_t i = 0; i < lines.size(); i++){
		float theta = lines[i][1];
		float temp = theta / CV_PI * 180;
		//cout << lines[i][0] << "   " << temp << endl;
		if((temp < 50) && (temp > 0)){
			angle += temp;
			num++;
		} else if(temp > 90 && temp < 135){
			angle2 += temp;
			num2++;
		}
	}
	if(num != 0){
		angle = angle / num;
		angle = angle * CV_PI / 180;
	}

	if(num2 != 0){
		angle2 = angle2 / num2;
		angle2 = angle2 * CV_PI / 180;
	}
	cout << "angle: " << angle / CV_PI * 180 << endl;	
	cout << "angle2: " << angle2 / CV_PI * 180 << endl;
	cout << "hof OK!" << endl;
}	
