#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main(){
	Mat src = 
imread("/mnt/hgfs/USHARE/test_digit_origin/11.png", 0);	
	
	imshow("test", src);

	Mat canny, rect;
	vector<Vec2f> lines;
	Canny(src, canny, 3, 9);
	HoughLines(canny, lines, 1, CV_PI/180, 25);
	float angle = 0, angle2 = 0;
	int num = 0, num2 = 0;
	for(size_t i = 0; i < lines.size(); i++){
		float theta = lines[i][1];// radian
		float temp = theta / CV_PI * 180;// angle
		if((temp < 50) && (temp > 0)){
			angle += temp;
			num++;
		} else if((temp > 70) && (temp < 140)){
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

	cout << "angle: " << angle*180/CV_PI << endl;
	cout << "angle2: " << angle2*180/CV_PI << endl;
 
	return 0;
}
