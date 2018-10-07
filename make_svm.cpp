#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;

using namespace cv;
using namespace cv::ml;

Mat train_data;
Mat train_classes;
Mat test_data;

/**----------------------------------------------------------
  -- 10 --> d(small)       11 --> f        12 --> h        --
  -- 13 --> l              14 --> p        15 --> u        -- 
  -- 16 --> e              17 --> c                        --
  --													   -*/

/** train_data without preprocess */
/** train_data2 without preprocess(and a little tilt) */
/** train_data3 haven been preprocess */
/** test_data's digit without preprocess */
/** test_data's alphabet without preprocess */

void get_train_data(); /* 64 */
void get_train_data2();/* 83 */
void get_train_data3();/* 119 */
void preprocess(Mat &src);/* 136 */
void preprocess2(Mat &src);/* 172 */

int main(){		
	Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
 	svm->setKernel(SVM::LINEAR);// 2
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	get_train_data();
//	get_train_data2();
//  get_train_data3();
	
	train_data.convertTo(train_data, CV_32F);
	train_classes.convertTo(train_classes, CV_32S);
	
	svm->train(train_data, ROW_SAMPLE, train_classes);	
	
	Mat src = imread("./test_data/13.png", 0);	
	preprocess(src);
	preprocess2(src);
	
	if(!src.data){
		perror("read image failed\n");
	}
	test_data = src.reshape(0, 1);
	test_data.convertTo(test_data, CV_32F);
	
	auto r = svm->predict(test_data);
	cout << "result: "<< r << endl;
	return 0;
}

void get_train_data(){
	char path[255] = "./train_data";
	for(int i = 0; i <= 9; ++i){
		for(int j = 1; j <= 9; ++j){
			char file[255];
			sprintf(file, "%s/%d%d.png", path, i, j);
			Mat temp = imread(file, 0);
			preprocess(temp);
			preprocess2(temp);
			temp = temp.reshape(0, 1);
			temp.convertTo(temp, CV_32F);
			train_data.push_back(temp);
			train_classes.push_back(i);
		}
	}
}

void get_train_data2(){
	char path[255] = "./train_data2";
	for(int i = 0; i <= 9; ++i){
		for(int j = 1; j <= 19; ++j){
			char file[255];
			if(j < 10){
				sprintf(file, "%s/%d/0%d.jpg", path, i, j);
				Mat temp = imread(file, 0);
				preprocess(temp);
				preprocess2(temp);
				temp = temp.reshape(0, 1);
				temp.convertTo(temp, CV_32F);
				train_data.push_back(temp);
				train_classes.push_back(i);
			} else if(j >= 10) {
				sprintf(file, "%s/%d/%d.jpg", path, i, j);
				Mat temp = imread(file, 0);
				preprocess(temp);
				preprocess2(temp);
				temp = temp.reshape(0, 1);
				temp.convertTo(temp, CV_32F);
				train_data.push_back(temp);
				train_classes.push_back(i);
			}
		}
	}
}

void get_train_data3(){
	char path[255] = "./train_data3";
	for(int i = 0; i <= 7; ++i){
		for(int j = 1; j <= 9; ++j){
			char file[255];
			sprintf(file, "%s/%d%d.png", path, i, j);
			Mat temp = imread(file, 0);
			preprocess2(temp);
			temp = temp.reshape(0, 1);
			temp.convertTo(temp, CV_32F);
			train_data.push_back(temp);
			train_classes.push_back(i + 10);
		}
	}
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
/*
void foo(){
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
	//Mat src = imread("./test_data/74.jpg", 0);
	//Mat src = imread("./test_data/75.png", 0);
	//preprocess(src);
	
	//Mat src = imread("./test_data2/11.png", 0);
	//Mat src = imread("./test_data2/41.jpg", 0);
	//Mat src = imread("./test_data2/31.jpg", 0);
	//Mat src = imread("./test_data2/71.jpg", 0);
	//Mat src = imread("./test_data2/42.jpg", 0);
	Mat src = imread("./test_data2/32.jpg", 0);

	
	//Mat src = imread("./test_data/c1.jpg", 0);
	//Mat src = imread("./test_data/p1.jpg", 0);
	//Mat src = imread("./test_data/e1.jpg", 0);
	//Mat src = imread("./test_data/u1.jpg", 0);
	//Mat src = imread("./test_data/e2.jpg", 0);
	//Mat src = imread("./test_data/d1.jpg", 0);
	//Mat src = imread("./test_data/f1.jpg", 0);
}*/
