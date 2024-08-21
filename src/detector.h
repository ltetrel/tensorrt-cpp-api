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
        Detector(const std::filesystem::path& onnxModelPath, const std::filesystem::path& cfgPath);
        const std::vector<BoundingBox> mPredict(const cv::cuda::GpuMat& gpuImg);
        const std::vector<std::string>  mGetLabels();
        const std::vector<std::vector<float>> mGetColors();  //TODO: vector of cv::Scalar instead
        #ifdef WITH_BENCHMARK
            void mPrintBenchmarkSummary(int warmupSize = 3);
        #endif
    private:
        const std::vector<std::vector<cv::cuda::GpuMat>> mPreProcess(const cv::cuda::GpuMat& gpuImg);
        const std::vector<BoundingBox> mPostProcess(const std::vector<std::vector<std::vector<float>>>& features);
        void mSetNetSize(const cv::Size& size);
        void mSetImgSize(const cv::Size& size);

        Options aOptions;
        CfgParser aConfig;
        std::unique_ptr<Engine> aEngine = nullptr;
        cv::Size aNetSize;
        cv::Size aImgSize;
        #ifdef WITH_BENCHMARK
            std::vector<float> aPreTimeInMs, aInferTimeInMs, aPostTimeInMs;
        #endif
};
