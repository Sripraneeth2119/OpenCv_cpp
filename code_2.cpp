#include <chrono>
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
    Mat nearest;
    Mat linear;
    Mat cubic;

    // INTER_NEAREST
    auto start = chrono::steady_clock::now();

    for (int i = 0; i < 1000; ++i) {
        resize(img, nearest, new_size, 0, 0, cv::INTER_NEAREST);
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for 1000 iterations using INTER_NEAREST: " << duration.count() << " ms" << endl;

    // INTER_LINEAR
    auto start = chrono::steady_clock::now();

    for (int i = 0; i < 1000; ++i) {
        resize(img, linear, new_size, 0, 0, cv::INTER_LINEAR);
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for 1000 iterations using INTER_LINEAR: " << duration.count() << " ms" << endl;

    // INTER_CUBIC
    auto start = chrono::steady_clock::now();

    for (int i = 0; i < 1000; ++i) {
        resize(img, cubic, new_size, 0, 0, cv::INTER_CUBIC);
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for 1000 iterations using INTER_CUBIC: " << duration.count() << " ms" << endl;

    return 0;
}