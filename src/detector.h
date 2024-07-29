#pragma once

#include <numeric>

#include "engine.h"
#include "transforms.h"
#include "configParser.h"

class Detector{
    public:
        Detector(const std::filesystem::path onnxModelPath, const std::filesystem::path cfgPath);
        const CfgParser mGetConfig();
        const std::vector<AnnotItem> mPredict(const cv::cuda::GpuMat& gpuImg);  //TODO: should return a boundingBox class
    private:
        const std::vector<std::vector<cv::cuda::GpuMat>> mPreProcess(const cv::cuda::GpuMat& gpuImg);
        const std::vector<AnnotItem> mPostProcess(const std::vector<std::vector<std::vector<float>>>& features);
        CfgParser aConfig;
        Options aOptions;
        std::unique_ptr<Engine> aEngine = nullptr;
};
