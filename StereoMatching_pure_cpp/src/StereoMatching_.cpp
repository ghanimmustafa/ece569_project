#include "StereoMatching_.h"

using namespace std;
using namespace cv;




Mat
G2C(Mat disp,int param)
{
	Mat Hsv = Mat::zeros(disp.rows,disp.cols,CV_8UC3);
	for(int i = 0 ; i < disp.rows; i++)
		for(int j = 0; j < disp.cols; j++){
			if(disp.at<uchar>(i,j) != 0){
				Hsv.at<Vec3b>(i,j)[0] = disp.at<uchar>(i,j) * param;
				Hsv.at<Vec3b>(i,j)[1] = 255;
				Hsv.at<Vec3b>(i,j)[2] = 255;
			}
		}
	Mat dst;
	cvtColor(Hsv,dst,cv::COLOR_HSV2BGR_FULL);
	return dst;
}


void
Display(Mat *img,char* m)
{
	int key;
	if(img->empty())
		return;
	imwrite("../images/disp.jpg", *img);
	namedWindow( "window", WINDOW_NORMAL );
    imshow( "window", *img );
	for(;;){
		key = waitKey(0);
		if (key == -1)
			break;
	}
}

int
SAD(uint8_t* imgl, uint8_t* imgr, int plx, int ply, int prx, int pry, int w_size, int height, int width)
{

	int start = -(w_size )/2;
	int stop  = w_size -1;
	int sad = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++){
			sad += abs((int)imgl[((ply + j) * width + plx + i)] - (int)imgr[((pry + j) * width + prx + i)]);
		}
	return sad;
}

int
SSD(uint8_t* imgl, uint8_t* imgr, int plx, int ply, int prx, int pry,int w_size, int height, int width)
{
	int start = -(w_size )/2;
	int stop  = w_size -1;
	int ssd = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++)
			ssd += pow((int)imgl[((ply + j) * width + plx + i)] - (int)imgr[((pry + j) * width + prx + i)],2);
	return ssd;
}

int
getMin(int* row,int col, int height, int width)
{
	int vmin = row[col];
	int imin = 0;
	for(int i = 1; i < DISP ; i++){
		if (vmin > row[i * width + col]){
			vmin = row[i * width + col];
			imin = i;
		}
	}
	return imin;
}

Mat
crop(Mat img,Point2i p, int w_size)
{
    return img(Rect(p.x - w_size/2,p.y - w_size/2,w_size,w_size)).clone();
}

int
Ideal(Mat img,Point2i p)
{
	iW = 1;
    int window_size = 3;
    Scalar mean,stddev,bstddev;
	Mat b = crop(img,p,window_size);
    meanStdDev(b,mean,bstddev,cv::Mat());
    meanStdDev(b,mean,stddev,cv::Mat());
	b.release();
    while (abs(stddev[0] - bstddev[0]) <= 40/*25*/){
        if(window_size/2 >= p.x || window_size/2 >= p.y || window_size >= WINDOW_SIZE)
            return window_size;
        window_size += 2;
        Mat window = crop(img,p,window_size);
        meanStdDev(window,mean,stddev,cv::Mat());
        window.release();
    }
    return window_size - 2 ;
}

