#include "StereoMatching_.h"

unsigned int WINDOW_SIZE;
unsigned int DISP;
unsigned int iW;

using namespace std;
using namespace cv;

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

int
main(int argc,
	char** argv)
{
	unsigned int WINDOW_SIZE;

	Mat img1,img2,row;
	int (*Method)(uint8_t*, uint8_t*, int, int, int, int, int, int, int);

	if(argc < 6){
		cout <<"Usage : "<<argv[0]<<"  <image-file-name-1> <image-file-name-2> <Method> <WINDOW_SIZE> <DISP>"<<endl;
		exit(0);
	}

	if(string(argv[3]) == "SAD")
		Method = &SAD;

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

	//imwrite("grayscale.png", img1);

	if(img1.empty() || img2.empty() )
		return -1;

	//disp = Mat(img1.rows,img1.cols,CV_8UC1,Scalar::all(255));

    int height_right, width_right, depth_right;
    char right_image_path[100] = "../images/Art/view5.png";
    uint8_t* right_image = stbi_load(right_image_path, &width_right, &height_right, &depth_right, 1);
    if (right_image == NULL){
	std::cout << "Failed to open image " << right_image_path << std::endl;
        return -1;
    }
    int size_right = height_right * width_right;

	int height_left, width_left, depth_left;
    char left_image_path[100] = "../images/Art/view1.png";
    uint8_t* left_image = stbi_load(left_image_path, &width_left, &height_left, &depth_left, 1);
    if (left_image == NULL){
	std::cout << "Failed to open image " << left_image_path << std::endl;
        return -1;
    }
    int size_left = height_left * width_left;

	uint8_t* disp = (uint8_t *) malloc (sizeof (uint8_t) * (size_right));
	for(int i = 0; i < size_right; i++)disp[i] = 255;

	for(int l = WINDOW_SIZE/2; l < height_right - WINDOW_SIZE/2; l++){
		int* row = (int *) malloc (sizeof (int) * (DISP * width_right));
		for(int iter = 0; iter < DISP * width_right; iter++){
			row[iter] = 255;
		}
		for(int i = WINDOW_SIZE/2 ; i < width_right - WINDOW_SIZE/2; i++){
			for(int j = 0; j < DISP ; j++){
				if((i+j) >= (width_right - WINDOW_SIZE /2) )
					break;
				row[j * width_right + i] = Method(right_image,left_image, i, l, i+j, l, WINDOW_SIZE, height_right, width_right);
			}
		}
		for(int i = WINDOW_SIZE/2; i < width_right - WINDOW_SIZE/2 ; i++){
			disp[l * width_right + i] =  getMin(row, i, height_right, width_right);
		}	
	}
	stbi_write_png("grayscale_.png", width_right, height_right, 1, disp, width_right);
	return 0;
}
