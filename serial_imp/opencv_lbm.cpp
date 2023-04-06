#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // Read input images
    Mat img1 = imread("left.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("right.png", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        cout << "Error: could not read input images" << endl;
        return -1;
    }

    // Get block size and search range from command line arguments
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " block_size search_range" << endl;
        return -1;
    }
    int block_size = atoi(argv[1]);
    int search_range = atoi(argv[2]);

    // Compute maximum disparity
    int max_disparity = img1.cols - block_size;

    // Create output image
    Mat disp = Mat::zeros(img1.size(), CV_8U);

    // Compute disparity for each pixel in left image
    auto start_time = chrono::high_resolution_clock::now();
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            int best_x = x;
            int best_sad = INT_MAX;
            for (int dx = 0; dx < search_range; dx++) {
                if (x + dx >= max_disparity) {
                    break;
                }
                int cur_sad = 0;
                for (int i = 0; i < block_size; i++) {
                    for (int j = 0; j < block_size; j++) {
                        int px1 = img1.at<uchar>(y + i, x + j);
                        int px2 = img2.at<uchar>(y + i, x + dx + j);
                        cur_sad += abs(px1 - px2);
                    }
                }
                if (cur_sad < best_sad) {
                    best_x = x + dx;
                    best_sad = cur_sad;
                }
            }
            // Set disparity value based on best matching pixel
            disp.at<uchar>(y, x) = (best_x - x) * (255.0 / max_disparity);
        }
    }
    auto end_time = chrono::high_resolution_clock::now();

    // Normalize output image
    Mat disp_norm;
    normalize(disp, disp_norm, 0, 255, NORM_MINMAX, CV_8U);

    // Apply color map to disparity image
    Mat disp_color;
    applyColorMap(disp_norm, disp_color, COLORMAP_JET);

    // Display output image
    imshow("Disparity", disp_color);
    waitKey(0);

    // Save output image
    imwrite("disparity.png", disp_color);

    // Output timing information
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Execution time: " << elapsed_time << " ms" << endl;

    return 0;
}

