#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {

    cout << "Loading the Image" << endl;
    cv::Mat img = cv::imread("G178_2-1080.BMP");
    
    if(img.empty()) {
        cout << "Image is not being Opened" << endl;
        return -1;
    }

    // Define the New output sizes
    Size new_size(img.cols / 2, img.rows / 2);

    // INTER_NEAREST
    Mat resized_nearest;
    resize(img, resized_nearest, new_size, 0, 0, cv::INTER_NEAREST);

    // INTER_LINEAR
    Mat resized_linear;
    resize(img, resized_linear, new_size, 0, 0, cv::INTER_LINEAR);

    // INTER_CUBIC
    Mat resized_cubic;
    resize(img, resized_cubic, new_size, 0, 0, cv::INTER_CUBIC);

    // Display 
    imshow("Nearest Neighbor", resized_nearest);
    imshow("Linear Interpolation", resized_linear);
    imshow("Cubic Interpolation", resized_cubic);
    waitKey(0);

    return 0;
}