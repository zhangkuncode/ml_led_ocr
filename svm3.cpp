#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

using namespace cv;
using namespace cv::ml;

Mat train_data;
Mat train_classes;
Mat test_data; // background must be white

void set_test_data(Mat &src);
void get_train_data(); 
void preprocess(Mat &src);
void preprocess2(Mat &src);
void preprocess3(Mat &src);	

int main(){		
	Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
 	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	get_train_data();
	
	train_data.convertTo(train_data, CV_32F);
	train_classes.convertTo(train_classes, CV_32S);
	
	svm->train(train_data, ROW_SAMPLE, train_classes);	
	
	//Mat src = imread("./test_data/43.jpg", 0);
	//Mat src = imread("./test_data/24.jpg", 0);
	//Mat src = imread("./test_data/05.jpg", 0);
	//Mat src = imread("./test_data/71.jpg", 0);
	//Mat src = imread("./test_data/34.jpg", 0);
	//Mat src = imread("./test_data/15.jpg", 0);
	//Mat src = imread("./test_data/23.jpg", 0);
	//Mat src = imread("./test_data/33.jpg", 0);
	//Mat src = imread("./test_data/42.jpg", 0);
	//Mat src = imread("./test_data/51.jpg", 0);
	//Mat src = imread("./test_data/61.jpg", 0);
	//Mat src = imread("./test_data/76.jpg", 0);
	//Mat src = imread("./test_data/81.jpg", 0);
	//Mat src = imread("./test_data/14.png", 0);
	//Mat src = imread("./test_data/13.png", 0);
	//Mat src = imread("./test_data/22.png", 0);
	//Mat src = imread("./test_data/04.png", 0);
	//Mat src = imread("./test_data/11.png", 0);
	//Mat src = imread("./test_data/32.png", 0);
	//Mat src = imread("./test_data/03.png", 0);
	//Mat src = imread("./test_data/91.png", 0);
	//Mat src = imread("./test_data/41.jpg", 0);
	//Mat src = imread("./test_data/31.jpg", 0);
	//Mat src = imread("./test_data/21.jpg", 0);
	//Mat src = imread("./test_data/01.jpg", 0);
	//Mat src = imread("./test_data/02.jpg", 0);
	//Mat src = imread("./test_data/12.png", 0);
	//Mat src = imread("./test_data/53.png", 0);
	//Mat src = imread("./test_data/52.jpg", 0);
	//Mat src = imread("./test_data/72.jpg", 0);
	//Mat src = imread("./test_data/73.jpg", 0);
	Mat src = imread("./test_data/74.jpg", 0);
	//Mat src = imread("./test_data/75.png", 0);
	if(!src.data){
		perror("read image failed\n");
	}
	src = src < 100;
	set_test_data(src);

	test_data = src.reshape(0, 1);
	test_data.convertTo(test_data, CV_32F);
	
	auto r = svm->predict(test_data);
	cout << "result: "<< r << endl;
	
	waitKey(0);
	return 0;
}

void set_test_data(Mat &src){
	imshow("origin", src);
	preprocess3(src);
	preprocess(src);
	imshow("done", src);
	preprocess2(src);
}

void get_train_data(){
	char path[255] = "./train_data";
	for(int i = 0; i <= 9; ++i){
		for(int j = 1; j <= 9; ++j){
			char file[255];
			sprintf(file, "%s/%d%d.png", path, i, j);
			Mat temp = imread(file, 0);
			temp = temp < 100;
			preprocess(temp);
			preprocess2(temp);
			temp = temp.reshape(0, 1);
			temp.convertTo(temp, CV_32F);
			train_data.push_back(temp);
			train_classes.push_back(i);
		}
	}
	cout << "get data OK" << endl;
}

/** input data background must be black */
void preprocess(Mat &src){
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

/** Dimensionality reduction for input-features */
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

void preprocess3(Mat &src){
	Mat canny, rect;
	vector<Vec2f> lines;
	Canny(src, canny, 3, 9);
	HoughLines(canny, lines, 1, CV_PI/180, 25);
	float angle = 0, angle2 = 0;
	int num = 0, num2 = 0;
	for(size_t i = 0; i < lines.size(); i++){
		float theta = lines[i][1];
		float temp = theta / CV_PI * 180;
		if((temp < 50) && (temp > 0)){
			angle += temp;
			num++;
		} else if((temp > 90) && (temp < 140)){
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
	Mat out;
	warpAffine(src, out, rect, src.size());	
	src = out.clone();
}
