#include "StereoMatching.h"
#include <math.h>

using namespace std;

int SAD(uint8_t* imgl, uint8_t* imgr, int plx, int ply, int prx, int pry, int w_size, int height, int width)
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

int SSD(uint8_t* imgl, uint8_t* imgr, int plx, int ply, int prx, int pry,int w_size, int height, int width)
{
	int start = -(w_size )/2;
	int stop  = w_size -1;
	int ssd = 0;
	for(int i = start; i < stop; i++)
		for(int j = start; j < stop; j++)
			ssd += pow((int)imgl[((ply + j) * width + plx + i)] - (int)imgr[((pry + j) * width + prx + i)],2);
	return ssd;
}

int getMin(int* row,int col, int height, int width)
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



