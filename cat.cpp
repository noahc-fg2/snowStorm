#include "cat.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/video.hpp>
#include <opencv4/opencv2/objdetect.hpp>
#include <onnx/onnxruntime_cxx_api.h>



cat::state cat::detect(const cv::Mat& frame) {
    cv::Mat frameTL;
    cv::Mat frameTR;
    cv::Mat frameBL;
    cv::Mat frameBR;
    cv::Mat frameC;

    // Split the frame into four quadrants
    int halfHeight = (frame.rows / 2) ;
    int halfWidth = (frame.cols / 2) ;

    frameTL = frame(cv::Rect(0, 0, halfWidth, halfHeight));
    frameTR = frame(cv::Rect(halfWidth, 0, halfWidth, halfHeight));
    frameBL = frame(cv::Rect(0, halfHeight, halfWidth, halfHeight));
    frameBR = frame(cv::Rect(halfWidth, halfHeight, halfWidth, halfHeight));
    frameC = frame(cv::Rect(halfWidth/2, halfHeight/2, halfWidth, halfHeight));

}


int cat::getX(std::vector<cat::detection> detections) {

}

int cat::getY(std::vector<cat::detection> detections) {

}

std::vector<cat::detection> cat::detectOnTile(const cv::Mat& subFrame, tile t) {

    std::vector<cat::detection> detections;
    // Implement the detection logic for the given tile here
    
    
    
    
    
    
    
    
    
    
    
    
    //End Detection Logic
    return detections;
}

cv::Mat cat::preprocessFrame(const cv::Mat& subFrame) {
    cv::Mat preprocessedFrame;
    cv::Mat processedFrame;
    float scale;

    scale = std::min(640.0f / subFrame.cols, 640.0f / subFrame.rows);

    int newHight = static_cast<int>(subFrame.rows * scale);
    int newWidth = static_cast<int>(subFrame.cols * scale);
    cv::resize(subFrame, preprocessedFrame, cv::Size(newWidth, newHight));

    // Implement the preprocessing logic here
    
    int offsetX = (640 - newWidth) / 2;
    int offsetY = (640 - newHight) / 2;
    preprocessedFrame.copyTo(processedFrame(cv::Rect(offsetX, offsetY, newWidth, newHight)));

    cv::copyMakeBorder(preprocessedFrame, preprocessedFrame, offsetY, offsetY, offsetX, offsetX, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return processedFrame;
}