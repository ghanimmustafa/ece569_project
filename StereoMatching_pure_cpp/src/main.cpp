#include "StereoMatching.h"

unsigned int WINDOW_SIZE;
unsigned int DISP;
unsigned int iW;

using namespace std;
using namespace cv;

int
main(int argc,
	char** argv)
{
	unsigned int WINDOW_SIZE;

	Mat img1,img2,row,disp;
	int (*Method)(Mat,Mat,Point2i,Point2i,int);

	if(argc < 6){
		cout <<"Usage : "<<argv[0]<<"  <image-file-name-1> <image-file-name-2> <Method> <WINDOW_SIZE> <DISP>"<<endl;
		exit(0);
	}

	if(string(argv[3]) == "SAD")
		Method = &SAD;

	else if(string(argv[3]) == "LBP")
		Method = &LBP;

	else if(string(argv[3]) == "SSD")
		Method = &SSD;
	
	else{
		cout << "Unknown Method" << endl;
		exit(0);
	}

	WINDOW_SIZE = atoi(argv[4]);
	DISP 		= atoi(argv[5]);

	img1 = imread(argv[2],IMREAD_GRAYSCALE);
	img2 = imread(argv[1],IMREAD_GRAYSCALE);

	if(img1.empty() || img2.empty() )
		return -1;

	disp = Mat(img1.rows,img1.cols,CV_8UC1,Scalar::all(255));

	for(int l = WINDOW_SIZE/2; l < img1.rows - WINDOW_SIZE/2; l++){
		row = Mat(DISP,img1.cols,CV_32S,Scalar::all(255));
		for(int i = WINDOW_SIZE/2 ; i < img1.cols - WINDOW_SIZE/2; i++){
			for(int j = 0; j < DISP ; j++){
				if((i+j) >= (img1.cols - WINDOW_SIZE /2) )
					break;
				row.at<int>(j,i) = Method(img1,img2,Point2i(i,l),Point2i(i+j,l), WINDOW_SIZE/*Ideal(img1,Point2i(i,l))*/);
			}
		}
		for(int i = WINDOW_SIZE/2; i < img1.cols - WINDOW_SIZE/2 ; i++){
			disp.at<uchar>(l,i) =  getMin(row,i);
		}
	}

	for(int i = 0; i < 200; i++)printf("%d ", disp.at<uchar>(120, i));
	printf("\n\n\n");

	medianBlur(disp,disp,3);

	for(int i = 0; i < 200; i++)printf("%d ", disp.at<uchar>(120, i));
	printf("\n\n\n");

	normalize(disp,disp,0,255,NORM_MINMAX,CV_8UC1);

	for(int i = 0; i < 200; i++)printf("%d ", disp.at<uchar>(120, i));
	printf("\n\n\n");

	//Mat dst = G2C(disp,4);
	//Mat gray;
	//cvtColor(dst, gray, COLOR_BGR2GRAY);
	Display(&disp,argv[3]);
	return 0;
}
