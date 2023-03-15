#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
// Compile: g++ opencv_test.cpp -o opencv_test `pkg-config --cflags --libs opencv4`
// Run: ./opencv_test 

int main(int argc, char** argv){

	cv::Mat image;
	image = cv::imread("lena.png", cv::IMREAD_UNCHANGED);
  // Convert to grayscale
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_RGB2GRAY);
	if(!image.data){
		std::cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", gray_image);

	cv::waitKey(0);
	return 0;
}
