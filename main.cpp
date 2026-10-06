#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include <iostream>
#include "cat.h"

int main() {
    int i =0;
    bool recording = false;
    cat snow;
    cv::VideoCapture cap(0, cv::CAP_V4L2); // Open the default camera
    cap.set(cv::CAP_PROP_FOURCC,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::VideoWriter writer(std::to_string(std::time(nullptr)) + ".mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1920, 1080));

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
        snow.detect(frame);

        if (cv::waitKey(1) == 27) break; // Press 'Esc' to exit
        if (cv::waitKey(1) == 'r') {
            recording = !recording;
            if (!recording) {
                i = 0;
            }
            else if (recording) {
                
                writer.write(frame);
            }
        }
    }
  
    cap.release(); // Release the camera
    cv::destroyAllWindows(); // Destroy all OpenCV windows
    if (writer.isOpened()) {
        writer.release();
    }
    
    return 0;
}   




