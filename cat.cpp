#include "cat.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/video.hpp>
#include <opencv4/opencv2/objdetect.hpp>
#include <algorithm>
#include <onnxruntime_cxx_api.h>


cat::cat()
    : env(ORT_LOGGING_LEVEL_WARNING, "CatDetector"),
      sessionOptions(),
      session(nullptr)
{
    sessionOptions.SetIntraOpNumThreads(4);

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;

    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

    const char* modelPath = "yolo11n.onnx";

    session = std::make_unique<Ort::Session>(
        env,
        modelPath,
        sessionOptions
    );
}
cat::state cat::detect(cv::Mat& frame)
{
    int halfHeight = frame.rows / 2;
    int halfWidth  = frame.cols / 2;

    cv::Mat frameTL = frame(cv::Rect(0, 0, halfWidth, halfHeight));
    cv::Mat frameTR = frame(cv::Rect(halfWidth, 0, halfWidth, halfHeight));
    cv::Mat frameBL = frame(cv::Rect(0, halfHeight, halfWidth, halfHeight));
    cv::Mat frameBR = frame(cv::Rect(halfWidth, halfHeight, halfWidth, halfHeight));
    cv::Mat frameC = frame(cv::Rect(halfWidth / 2, halfHeight / 2, halfWidth, halfHeight));

    std::array<cv::Mat, 5> tiles = { frameTL, frameTR, frameBL, frameBR, frameC };
    std::array<tile, 5> locations = {
        tile::TopLeft, tile::TopRight, tile::BottomLeft,
        tile::BottomRight, tile::Center
    };
    std::array<cv::Point, 5> origins = {
        cv::Point(0, 0),
        cv::Point(halfWidth, 0),
        cv::Point(0, halfHeight),
        cv::Point(halfWidth, halfHeight),
        cv::Point(halfWidth / 2, halfHeight / 2)
    };

    constexpr int batchSize = 5;
    constexpr int channels = 3;
    constexpr int inputWidth = 640;
    constexpr int inputHeight = 640;
    constexpr int imageSize = channels * inputWidth * inputHeight;
    constexpr int numCandidates = 8400;
    constexpr int numClasses = 80;
    constexpr int outputChannels = 4 + numClasses;

    std::array<PreprocessedFrame, batchSize> prepared;
    std::vector<float> inputTensorValues(batchSize * imageSize);

    for (int n = 0; n < batchSize; ++n)
    {
        prepared[n] = preprocessFrame(tiles[n]);
        const cv::Mat& image = prepared[n].image;
        float* dst = inputTensorValues.data() + n * imageSize;

        for (int y = 0; y < inputHeight; ++y)
        {
            const cv::Vec3f* row = image.ptr<cv::Vec3f>(y);
            for (int x = 0; x < inputWidth; ++x)
            {
                dst[0 * inputWidth * inputHeight + y * inputWidth + x] = row[x][0];
                dst[1 * inputWidth * inputHeight + y * inputWidth + x] = row[x][1];
                dst[2 * inputWidth * inputHeight + y * inputWidth + x] = row[x][2];
            }
        }
    }

    std::array<int64_t, 4> inputShape = { batchSize, channels, inputHeight, inputWidth };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputShape.data(),
        inputShape.size());

    const char* inputNames[] = { "images" };
    const char* outputNames[] = { "output0" };

    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1);

    float* output = outputTensors[0].GetTensorMutableData<float>();
    std::vector<detection> detections;

    for (int n = 0; n < batchSize; ++n)
    {
        const size_t batchOffset =
            static_cast<size_t>(n) * outputChannels * numCandidates;

        for (int i = 0; i < numCandidates; ++i)
        {
            float cx = output[batchOffset + 0 * numCandidates + i];
            float cy = output[batchOffset + 1 * numCandidates + i];
            float width = output[batchOffset + 2 * numCandidates + i];
            float height = output[batchOffset + 3 * numCandidates + i];
            float confidence = output[
                batchOffset + (4 + 15) * numCandidates + i];

            if (confidence < 0.20f)
                continue;

            float x1 = (cx - width / 2.0f - prepared[n].offsetX) / prepared[n].scale;
            float y1 = (cy - height / 2.0f - prepared[n].offsetY) / prepared[n].scale;
            float x2 = (cx + width / 2.0f - prepared[n].offsetX) / prepared[n].scale;
            float y2 = (cy + height / 2.0f - prepared[n].offsetY) / prepared[n].scale;

            cv::Rect box(
                static_cast<int>(x1),
                static_cast<int>(y1),
                static_cast<int>(x2 - x1),
                static_cast<int>(y2 - y1));

            box &= cv::Rect(0, 0, tiles[n].cols, tiles[n].rows);
            if (box.area() <= 0)
                continue;

            detection d;
            d.classID = 15;
            d.box = box;
            d.confidence = confidence;
            d.location = locations[n];
            d.box.x += origins[n].x;
            d.box.y += origins[n].y;
            detections.push_back(d);
        }
    }

    for (const auto& d : detections)
    {
        cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 3);
        std::string label = "Cat " + std::to_string(d.confidence);
        cv::putText(frame, label,
            cv::Point(d.box.x, d.box.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.7,
            cv::Scalar(0, 255, 0), 2);
    }

    return detections.empty() ? state::NotFound : state::Found;
}


int cat::getX(std::vector<cat::detection> detections) {

    return 0;
}

int cat::getY(std::vector<cat::detection> detections) {

    return 0;
}

std::vector<cat::detection> cat::detectOnTile(
    const cv::Mat& subFrame)
{
    std::vector<cat::detection> detections;

    // -------------------------
    // Preprocess
    // -------------------------

    cat::PreprocessedFrame prepared =
        preprocessFrame(subFrame);

    cv::Mat preparedFrame = prepared.image;

    // -------------------------
    // Convert HWC -> CHW
    // -------------------------

    std::vector<float> inputTensorValues(
        3 * 640 * 640
    );

    for (int y = 0; y < 640; y++)
    {
        for (int x = 0; x < 640; x++)
        {
            cv::Vec3f pixel =
                preparedFrame.at<cv::Vec3f>(y, x);

            inputTensorValues[
                0 * 640 * 640 +
                y * 640 +
                x
            ] = pixel[0];

            inputTensorValues[
                1 * 640 * 640 +
                y * 640 +
                x
            ] = pixel[1];

            inputTensorValues[
                2 * 640 * 640 +
                y * 640 +
                x
            ] = pixel[2];
        }
    }

    // -------------------------
    // ONNX input tensor
    // -------------------------

    std::array<int64_t, 4> inputShape = {
        1, 3, 640, 640
    };

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );

    // -------------------------
    // Run YOLO
    // -------------------------

    const char* inputNames[] = {
        "images"
    };

    const char* outputNames[] = {
        "output0"
    };

    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    // -------------------------
    // Get YOLO output
    // -------------------------

    float* output =
        outputTensors[0].GetTensorMutableData<float>();

    constexpr int numCandidates = 8400;
    constexpr int numClasses = 80;

    // -------------------------
    // Decode detections
    // -------------------------

    for (int i = 0; i < numCandidates; i++)
    {
        float cx =
            output[0 * numCandidates + i];

        float cy =
            output[1 * numCandidates + i];

        float width =
            output[2 * numCandidates + i];

        float height =
            output[3 * numCandidates + i];

        // COCO class 15 = cat
        float confidence =
            output[(4 + 15) * numCandidates + i];

        std::cout<< "Cat confidence: "<< confidence<< std::endl;
        if (confidence < 0.20f)
            continue;

        // YOLO coordinates are relative
        // to the 640x640 letterboxed image.
        float x1 =
            (cx - width / 2.0f
             - prepared.offsetX)
            / prepared.scale;

        float y1 =
            (cy - height / 2.0f
             - prepared.offsetY)
            / prepared.scale;

        float x2 =
            (cx + width / 2.0f
             - prepared.offsetX)
            / prepared.scale;

        float y2 =
            (cy + height / 2.0f
             - prepared.offsetY)
            / prepared.scale;

        int left =
            static_cast<int>(x1);

        int top =
            static_cast<int>(y1);

        int right =
            static_cast<int>(x2);

        int bottom =
            static_cast<int>(y2);

        cv::Rect box(
            left,
            top,
            right - left,
            bottom - top
        );

        // Keep the detection inside
        // the local tile.
        box &= cv::Rect(
            0,
            0,
            subFrame.cols,
            subFrame.rows
        );

        if (box.area() <= 0)
            continue;

        cat::detection detection;

        detection.classID = 15;
        detection.box = box;
        detection.confidence = confidence;

        detections.push_back(detection);
    }

    return detections;
}
   
cat::PreprocessedFrame cat::preprocessFrame(const cv::Mat& subFrame){
    PreprocessedFrame result;

    result.scale = std::min(
        640.0f / subFrame.cols,
        640.0f / subFrame.rows
    );

    const int newWidth = static_cast<int>(
        subFrame.cols * result.scale
    );

    const int newHeight = static_cast<int>(
        subFrame.rows * result.scale
    );

    result.offsetX = (640 - newWidth) / 2;
    result.offsetY = (640 - newHeight) / 2;

    cv::Mat resizedFrame;
    cv::resize(
        subFrame,
        resizedFrame,
        cv::Size(newWidth, newHeight)
    );

    // Initialize the entire canvas before copying into it.
    cv::Mat letterboxed(
        640, 640, CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    resizedFrame.copyTo(letterboxed(cv::Rect(
        result.offsetX,
        result.offsetY,
        newWidth,
        newHeight
    )));

    cv::Mat rgbFrame;
    cv::cvtColor(
        letterboxed,
        rgbFrame,
        cv::COLOR_BGR2RGB
    );

    rgbFrame.convertTo(
        result.image,
        CV_32FC3,
        1.0 / 255.0
    );

    return result;
}
