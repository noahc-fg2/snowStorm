#ifndef CAT_H
#define CAT_H
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <memory>

class cat{


    public:

        cat();
        ~cat();

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

        struct PreprocessedFrame {
            cv::Mat image;
            float scale;
            int offsetX;
            int offsetY;
        };

        cat::tile detect(cv::Mat& frame, cat::tile targetTile);

    private:
        
        int getX(std::vector<cat::detection> detections);
        int getY(std::vector<cat::detection> detections);
        std::vector<detection> detectOnTile(const cv::Mat& subFrame);
        cat::PreprocessedFrame preprocessFrame(const cv::Mat& frame);


        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "snowStorm"};
        Ort::SessionOptions sessionOptions;
        std::unique_ptr<Ort::Session> session;


};

#endif // CAT_H