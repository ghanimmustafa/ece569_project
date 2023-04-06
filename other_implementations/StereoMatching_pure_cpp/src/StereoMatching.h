#ifndef _STEREO_
#define _STEREO_

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

int SAD(uint8_t*, uint8_t*, int, int, int, int, int, int, int);
int SSD(uint8_t*, uint8_t*, int, int, int, int, int, int, int);
int getMin(int*,int, int, int);

extern unsigned int WINDOW_SIZE;
extern unsigned int DISP;
extern unsigned int BETA;
extern unsigned int iW;

extern unsigned int gDisp_dim;
extern unsigned int gNc;

#endif
