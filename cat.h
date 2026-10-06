#ifndef CAT_H
#define CAT_H
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <onnx/onnxruntime_cxx_api.h>

class cat{


    public:
        enum class tile {
            TopLeft,
            TopRight,
            BottomLeft,
            BottomRight,
            Center
        };

        enum class state {
            Found,
            Moving,
            Sleeping,
            NotFound
        };
    

        struct detection {
            int classID;
            cv::Rect box;
            float confidence;
            tile location;
        };

        state detect(const cv::Mat& frame);

    private:
        
        int getX(std::vector<cat::detection> detections);
        int getY(std::vector<cat::detection> detections);
        std::vector<detection> detectOnTile(const cv::Mat& subFrame, tile t);
        cv::Mat preprocessFrame(const cv::Mat& frame);


        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "snowStorm"};
        Ort::SessionOptions sessionOptions;
        std::unique_ptr<Ort::Session> session;


};

#endif // CAT_H