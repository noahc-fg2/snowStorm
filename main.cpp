#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include <iostream>
#include "cat.h"

int main() {
    cat snow;
    cv::VideoCapture cap(0, cv::CAP_V4L2); // Open the default camera
    cap.set(cv::CAP_PROP_FOURCC,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }
    cv::namedWindow("Snow_Storm", cv::WINDOW_NORMAL);
    while (true) {
        cv::Mat frame;
        cap >> frame; // Capture a new frame
        if (frame.empty()) break;
        cv::imshow("Snow_Storm", frame);
        if (cv::waitKey(1) == 27) break; // Press 'Esc' to exit
    }
    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Destroy all OpenCV windows
    
    
    return 0;
}   




