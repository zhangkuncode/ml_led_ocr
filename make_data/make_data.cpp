#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
	char path[255] = "./origin_data";
	char path2[255] = "./train_data";
	for(int i = 0; i < 10; ++i){
		char file[255];
		char file2[255];
		sprintf(file, "%s/%d.png", path, i);
		//cout << file << endl;
		Mat src = imread(file, 0);
		if(!src.data){
			perror("read file filed!\n");
		}
		resize(src, src, Size(128, 128));
		threshold(src, src, 80, 255, THRESH_BINARY);
		src = src < 70;
		medianBlur(src, src, 7);
		src.convertTo(src, CV_32F);
		sprintf(file2, "%s/%d1.png", path2, i);
		imwrite(file2, src);
		//imshow(file, src);	
	}
	return 0;
}

