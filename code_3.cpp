#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

void Resize_Nearest(const Mat& src, Mat& dst, int new_width, int new_height) {
    dst.create(new_height, new_width, src.type());
    float scale_x = (float)src.cols / new_width;
    float scale_y = (float)src.rows / new_height;
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int src_x = cvRound(x * scale_x);
            int src_y = cvRound(y * scale_y);
            dst.at<Vec3b>(y, x) = src.at<Vec3b>(src_y, src_x);
        }
    }
}

void Resize_Linear(const Mat& src, Mat& dst, int new_width, int new_height) {
    dst.create(new_height, new_width, src.type());
    float scale_x = (float)(src.cols - 1) / new_width;
    float scale_y = (float)(src.rows - 1) / new_height;
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = std::min(x0 + 1, src.cols - 1);
            int y1 = std::min(y0 + 1, src.rows - 1);
            float dx = src_x - x0;
            float dy = src_y - y0;
            Vec3b p00 = src.at<Vec3b>(y0, x0);
            Vec3b p01 = src.at<Vec3b>(y0, x1);
            Vec3b p10 = src.at<Vec3b>(y1, x0);
            Vec3b p11 = src.at<Vec3b>(y1, x1);
            Vec3f interpolated_value = (1.0f - dx) * (1.0f - dy) * p00
                + dx * (1.0f - dy) * p01
                + (1.0f - dx) * dy * p10
                + dx * dy * p11;
            dst.at<cv::Vec3b>(y, x) = interpolated_value;
        }
    }
}

void Resize_Cubic(const Mat& src, Mat& dst, int new_width, int new_height) {
    dst.create(new_height, new_width, src.type());
    float scale_x = (float)(src.cols - 1) / new_width;
    float scale_y = (float)(src.rows - 1) / new_height;
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            int x0 = (int)src_x - 1;
            int y0 = (int)src_y - 1;
            int x1 = min(x0 + 1, src.cols - 1);
            int y1 = min(y0 + 1, src.rows - 1);
            int x2 = min(x0 + 2, src.cols - 1);
            int y2 = min(y0 + 2, src.rows - 1);
            int x3 = min(x0 + 3, src.cols - 1);
            int y3 = min(y0 + 3, src.rows - 1);
            float dx = src_x - x0;
            float dy = src_y - y0;
            float dx2 = dx * dx;
            float dx3 = dx2 * dx;
            float dy2 = dy * dy;
            float dy3 = dy2 * dy;
            Vec3b p00 = src.at<Vec3b>(y0, x0);
            Vec3b p01 = src.at<Vec3b>(y0, x1);
            Vec3b p02 = src.at<Vec3b>(y0, x2);
            Vec3b p03 = src.at<Vec3b>(y0, x3);
            Vec3b p10 = src.at<Vec3b>(y1, x0);
            Vec3b p11 = src.at<Vec3b>(y1, x1);
            Vec3b p12 = src.at<Vec3b>(y1, x2);
            Vec3b p13 = src.at<Vec3b>(y1, x3);
            Vec3b p20 = src.at<Vec3b>(y2, x0);
            Vec3b p21 = src.at<Vec3b>(y2, x1);
            Vec3b p22 = src.at<Vec3b>(y2, x2);
            Vec3b p23 = src.at<Vec3b>(y2, x3);
            Vec3b p30 = src.at<Vec3b>(y3, x0);
            Vec3b p31 = src.at<Vec3b>(y3, x1);
            Vec3b p32 = src.at<Vec3b>(y3, x2);
            Vec3b p33 = src.at<Vec3b>(y3, x3);
            Vec3f interpolated_value = (1.0f / 6.0f) * (
                -p00 + 3.0f * p01 - 3.0f * p02 + p03 +
                3.0f * p10 - 9.0f * p11 + 9.0f * p12 - 3.0f * p13 +
                3.0f * p20 - 9.0f * p21 + 9.0f * p22 - 3.0f * p23 +
                -p30 + 3.0f * p31 - 3.0f * p32 + p33
                ) * dx3 +
                (1.0f / 6.0f) * (
                -p00 + 3.0f * p10 - 3.0f * p20 + p30 +
                3.0f * p01 - 9.0f * p11 + 9.0f * p21 - 3.0f * p31 +
                3.0f * p02 - 9.0f * p12 + 9.0f * p22 - 3.0f * p32 +
                -p03 + 3.0f * p13 - 3.0f * p23 + p33
                ) * dy3;
            dst.at<cv::Vec3b>(y, x) = interpolated_value;
        }
    }
}

int main() {
    Mat input_image = cv::imread("G178_2-1080.BMP");
    if(input_image.empty()) {
        cerr << "Image is not being Opened." << endl;
        return -1;
    }

    int new_width = input_image.cols / 2;
    int new_height = input_image.rows / 2;

    Mat nearest, linear, cubic;

    auto start = chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        Resize_Nearest(input_image, nearest, new_width, new_height);
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for 1000 iterations using INTER_NEAREST: " << duration.count() << " ms" << endl;

    auto start = chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        Resize_Linear(input_image, linear, new_width, new_height);
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for 1000 iterations using INTER_LINEAR: " << duration.count() << " ms" << endl;

    auto start = chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        Resize_Cubic(input_image, cubic, new_width, new_height);
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for 1000 iterations using INTER_CUBIC: " << duration.count() << " ms" << endl;

    return 0;
}
