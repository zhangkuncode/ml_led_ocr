#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

void preprocess(Mat &src);
void preprocess2(Mat &src);
void preprocess3(Mat &src);
void preprocess4(Mat &src);

int main(){
	//Mat src1 = imread("./test_data/01.jpg", 0);
	//Mat src1 = imread("./test_data/33.jpg", 0);
	Mat src1 = imread("./test_data/13.png", 0);
	if(!src1.data){
		perror("1 read file failed!\n");
	}
	src1 = src1 < 100;
	preprocess4(src1);

	waitKey(0);
	return 0;
}

void preprocess4(Mat &src){
	imshow("origin", src);

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

	cout << "angle: "  << angle / CV_PI * 180 << endl;
	cout << "angle2: " << angle2 / CV_PI * 180 << endl;
	
	float x_cdt = 40, y_cdt = 40, rate = 20;
	Point2f pdst[3] = {Point2f(x_cdt, y_cdt), Point2f(x_cdt, y_cdt-rate), 
					   Point2f(x_cdt+rate, y_cdt)};
	if(num2 != 0){
		Point2f psrc[3] = {Point2f(x_cdt, y_cdt), 
					       Point2f(x_cdt+rate*sin(angle),y_cdt-rate*cos(angle)),
					       Point2f(x_cdt+rate*sin(CV_PI-angle2),
								   y_cdt+rate*cos(CV_PI-angle2))};
		rect = getAffineTransform(psrc, pdst);
	} else {
		Point2f psrc[3] = {Point2f(x_cdt, y_cdt), 
						   Point2f(x_cdt+rate*sin(angle),y_cdt-rate*cos(angle)),
						   Point2f(x_cdt+rate,y_cdt)};	
		rect = getAffineTransform(psrc, pdst);
	}
/*	circle(canny, Point2f(x_cdt, y_cdt), 2, Scalar(255, 255, 255));
	circle(canny, Point2f(x_cdt+rate*sin(angle),
						  y_cdt-rate*cos(angle)), 5, Scalar(255, 255, 255));
	circle(canny, Point2f(x_cdt+rate*sin(CV_PI-angle2),
						  y_cdt+rate*cos(CV_PI-angle2)), 10, Scalar(255, 255, 255));
	cout << "angle: " << angle << endl;	
	cout << "angle2: " << angle2 << endl;	
	cout << "sin(angle): " << sin(angle) << endl;
	cout << "cos(angle): " << cos(angle) << endl;
	cout << "sin(180-angle2): " << sin(180-angle2) << endl;
	cout << "cos(180-angle2)" << cos(180-angle2) << endl;
	cout << "p1: " << x_cdt << " , " << y_cdt << endl;
	cout << "p2: " << x_cdt+rate*sin(angle) << " , " << y_cdt-rate*cos(angle) << endl;
	cout << "p3: " << x_cdt+rate*sin(180-angle2) << " , " << y_cdt+rate*cos(180-angle2) << endl;
*/
	imshow("canny", canny);
	/*} else {
		cout << "rotation" << endl;
		rect = getRotationMatrix2D(Point(src.cols/2, src.rows/2), angle/CV_PI*180, 1.0);
	}*/
	warpAffine(src, out, rect, src.size());	
	imshow("done", out);
}

void preprocess3(Mat &src){
	// change background to black
	src = src < 100;
	// finds edges
	Mat canny;
	vector<Vec2f> lines;
	Canny(src, canny, 3, 9);
	HoughLines(canny, lines, 1, CV_PI/180, 20);
	float angle = 0;
	int num = 0;
	for(size_t i = 0; i < lines.size(); i++){
		float theta = lines[i][1];
		if(((theta/CV_PI*180) < 90) && ((theta/CV_PI*180) > -90)){
			angle += theta/CV_PI*180;
			num++;
		}
	}
	if(num != 0){
		angle = angle / num;
	}
	Mat rect = getRotationMatrix2D(Point(32, 64), 17, 1.0);
	warpAffine(src, src, rect, src.size());
	
	preprocess(src);
}

void preprocess2(Mat &src){
	Scalar ss;
	for(int i = 0; i < (src.rows / 3); ++i){
		ss += sum(src.rowRange(i, i+1));
	}
	
	Scalar ss1;
	for(int i = (src.rows / 3); i < (src.rows / 3) * 2; ++i){
		ss1 += sum(src.rowRange(i, i+1));
	}

	Scalar ss2;
	for(int i = (src.rows / 3) * 2; i < src.rows; ++i){
		ss2 += sum(src.rowRange(i, i+1));
	}
	
	Mat o1 = src(Rect(0, 0, 32, 64));
	Scalar ss3;
	for(int i = 0; i < o1.rows; ++i){
		ss3 += sum(o1.rowRange(i, i+1));
	}
	
	Mat o2 = src(Rect(32, 0, 32, 64));
	Scalar ss4;
	for(int i = 0; i < o2.rows; ++i){
		ss4 += sum(o2.rowRange(i, i+1));
	}
	
	Mat o3 = src(Rect(0, 64, 32, 64));
	Scalar ss5;
	for(int i = 0; i < o3.rows; ++i){
		ss5 += sum(o3.rowRange(i, i+1));
	}

	Mat o4 = src(Rect(32, 64, 32, 64));
	Scalar ss6;
	for(int i = 0; i < o4.rows; ++i){
		ss6 += sum(o4.rowRange(i, i+1));
	}
	
	double dd[7] = {ss.val[0],  ss1.val[0], ss2.val[0], 
		            ss3.val[0], ss4.val[0], ss5.val[0], ss6.val[0]};
	double max = dd[0];
	for(int i = 1; i < 7; ++i){
		if(dd[i] > dd[i - 1]){
			max = dd[i];
		}
	}
	float arr[7][1] ={static_cast<float>(ss.val[0]  / max),  
		              static_cast<float>(ss1.val[0] / max), 
					  static_cast<float>(ss2.val[0] / max), 
					  static_cast<float>(ss3.val[0] / max), 
		              static_cast<float>(ss4.val[0] / max), 
					  static_cast<float>(ss5.val[0] / max), 
					  static_cast<float>(ss6.val[0] / max)} ;
	Mat out = Mat(7, 1, CV_32FC1, arr); 
	src = out.clone();
}

void preprocess(Mat &src){
	src = src < 100;
	int x_min = 0, y_min = 0, x_max = 0, y_max = 0;
	Scalar max = Scalar(src.cols * 255);
	Mat data;
	// find x
	int xminFound = 0;
	for(int i = 0; i < src.cols; ++i){
		data = src.colRange(i, i + 1);
		Scalar sss = sum(data);
		if(sss.val[0] > 0){
			x_max = i;
			if(!xminFound){
				x_min = i;
				xminFound = 1;
			}
		}
	}
	// find y
	int yminFound = 0;
	for(int i = 0; i < src.rows; ++i){
		data = src.rowRange(i, i + 1);
		Scalar sss = sum(data);
		if(sss.val[0] > 0){
			y_max = i;
			if(!yminFound){
				y_min = i;
				yminFound = 1;
			}
		}
	}
	int height = y_max - y_min;
	int weight = x_max - x_min;
	if((weight < (height / 2)) && ((height / 2) < (src.cols - x_min))){
		src = src(Rect(x_min, y_min, height / 2, height));
		src = src.clone();
		resize(src, src, Size(64, 128));
	} else if((weight < (height / 2)) && ((height / 2) >= (src.cols - x_min))){
		src = src(Rect(x_min, y_min, src.cols - x_min, height));
		src = src.clone();
		resize(src, src, Size(64, 128));
	} else {
		src = src(Rect(x_min, y_min, weight, height));
		src = src.clone();
		resize(src, src, Size(64, 128));
	}
}

