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
const int K = 3;

void get_train_data();
void get_train_data2();
void preprocess(Mat &src);

int main(){	
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(K);
	knn->setIsClassifier(true);

	get_train_data();
	get_train_data2();
	
	train_data.convertTo(train_data, CV_32F);
	train_classes.convertTo(train_classes, CV_32S);
	
	knn->train(train_data, ROW_SAMPLE, train_classes);
	//Mat src = imread("./test_data/43.jpg", 0);
	//Mat src = imread("./test_data/24.jpg", 0);
	//Mat src = imread("./test_data/05.jpg", 0);
	//Mat src = imread("./test_data/71.jpg", 0);
	//Mat src = imread("./test_data/34.jpg", 0);
	//Mat src = imread("./test_data/15.jpg", 0);
	Mat src = imread("./test_data/23.jpg", 0);
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
	
	preprocess(src);

	if(!src.data){
		perror("read image failed\n");
	}
	test_data = src.reshape(0, 1);
	test_data.convertTo(test_data, CV_32F);

	auto r = knn->findNearest(test_data, K, noArray());
	cout<<"result: "<< r <<endl;
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
			temp = temp.reshape(0, 1);
			temp.convertTo(temp, CV_32F);
			train_data.push_back(temp);
			train_classes.push_back(i);
		}
	}
	cout << "read train data OK" << endl;
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
				temp = temp.reshape(0, 1);
				temp.convertTo(temp, CV_32F);
				train_data.push_back(temp);
				train_classes.push_back(i);
			} else if(j >= 10) {
				sprintf(file, "%s/%d/%d.jpg", path, i, j);
				Mat temp = imread(file, 0);
				preprocess(temp);
				temp = temp.reshape(0, 1);
				temp.convertTo(temp, CV_32F);
				train_data.push_back(temp);
				train_classes.push_back(i);
			}
		}
	}
	cout << "read train data2 OK" << endl;
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
	src = src(Rect(x_min, y_min, x_max - x_min, y_max - y_min));
	src = src.clone();
	resize(src, src, Size(128, 128));
}
