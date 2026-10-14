#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include <iostream>
#include "cat.h"

int main() {
    int i =0;
    bool recording = false;
    cat snow;
    
    cat::tile currentTile;
    cat::tile nextTile = cat::tile::TopLeft;
    
    cv::VideoCapture cap(0, cv::CAP_V4L2); // Open the default camera
    cap.set(cv::CAP_PROP_FOURCC,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);

    cv::VideoWriter writer(
    std::to_string(std::time(nullptr)) + ".mp4",
    cv::VideoWriter::fourcc('m','p','4','v'),
    30,
    cv::Size(1920, 1080)
    );

    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }
    cv::namedWindow("Snow Storm", cv::WINDOW_NORMAL);
    
    auto lastFrameTime = std::chrono::steady_clock::now();
    double fps = 0.0;
    
    while (true) {
        
        currentTile = nextTile;

        cv::Mat frame;
        cap >> frame; // Capture a new frame
        if (frame.empty()) break;
        
        nextTile = snow.detect(frame, currentTile);

        // Calculate FPS
        auto currentTime = std::chrono::steady_clock::now();

        double deltaTime =
            std::chrono::duration<double>(currentTime - lastFrameTime).count();

        lastFrameTime = currentTime;

        if (deltaTime > 0.0)
        {
            fps = 1.0 / deltaTime;
        }

        // Draw FPS on image
        std::string fpsText = "FPS: " + std::to_string(fps);

        cv::putText(
            frame,
            fpsText,
            cv::Point(30, 50),
            cv::FONT_HERSHEY_SIMPLEX,
            1.2,
            cv::Scalar(0, 255, 0),
            3
        );

        cv::imshow("Snow Storm", frame);

        int key = cv::waitKey(1);

        if (key == 27) break; // Press 'Esc' to exit
        if (key == 'r') {
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




