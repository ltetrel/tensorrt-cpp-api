#pragma once

#include <memory>
#include <filesystem>
#include <vector>
#include <opencv2/core/cuda.hpp>

#include "engine.h"
#include "boundingBox.h"
#include "configParser.h"

class Detector{
    public:
        Detector(const std::filesystem::path onnxModelPath, const std::filesystem::path cfgPath);
        const CfgParser mGetConfig();
        const std::vector<BoundingBox> mPredict(const cv::cuda::GpuMat& gpuImg);  //TODO: should return a boundingBox class
    private:
        const std::vector<std::vector<cv::cuda::GpuMat>> mPreProcess(const cv::cuda::GpuMat& gpuImg);
        const std::vector<BoundingBox> mPostProcess(const std::vector<std::vector<std::vector<float>>>& features);
        CfgParser aConfig;
        Options aOptions;
        std::unique_ptr<Engine> aEngine = nullptr;
        cv::Size aNetShape;
};
