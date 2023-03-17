#ifndef _STEREO_
#define _STEREO_

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void Display(Mat*,char*);
void Save(Mat*,char*);
int SAD(uint8_t*, uint8_t*, int, int, int, int, int, int, int);
int SSD(uint8_t*, uint8_t*, int, int, int, int, int, int, int);
int getMin(int*,int, int, int);
Mat  G2C(Mat,int);
int Ideal(Mat,Point2i);
Mat crop(Mat,Point2i,int);


extern unsigned int WINDOW_SIZE;
extern unsigned int DISP;
extern unsigned int BETA;
extern unsigned int iW;

extern unsigned int gDisp_dim;
extern unsigned int gNc;

#endif
