#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "engine.h"
#include "boundingBox.h"
#include "utils.h"


enum class ResizeMethod {
    maintain_ar,
    scale,
};

enum class ColorMode{
    // can be any of https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
    RGB,
    BGR,
    GRAY,
};

namespace {

struct TransformValueMapper{
    std::unordered_map<std::string, ColorMode> const colorMode = {
        {"RGB", ColorMode::RGB},
        {"BGR", ColorMode::BGR},
        {"GRAY", ColorMode::GRAY}
    };
    std::unordered_map<std::string, ResizeMethod> const resizeMethod = {
        {"maintain_ar", ResizeMethod::maintain_ar},
        {"scale", ResizeMethod::scale}
    };
    std::unordered_map<std::string, Precision> const imageType = {
        {"int", Precision::INT8},
        {"float", Precision::FP32}
    };
    std::unordered_map<std::string, BoxFormat> const boxFormat = {
        {"xyxy", BoxFormat::xyxy},
        {"cxcywh", BoxFormat::cxcywh},
        {"xywh", BoxFormat::xywh}
    };
};

template <typename T>
T mapFileNodeString(cv::FileNode ocvFn){
    std::string key = ocvFn.string();
    T value;
    TransformValueMapper transformValueMapper;
    std::unordered_map<std::string, T> mapperTable;

    if constexpr(std::is_same<T, ColorMode>::value){
        mapperTable = transformValueMapper.colorMode;
    }
    else if constexpr(std::is_same<T, ResizeMethod>::value){
        mapperTable = transformValueMapper.resizeMethod;
    }
    else if constexpr(std::is_same<T, Precision>::value){
        mapperTable = transformValueMapper.imageType;
    }
    else if constexpr(std::is_same<T, BoxFormat>::value){
        mapperTable = transformValueMapper.boxFormat;
    }
    value = Utils::getValueFromMapKey<T>(mapperTable, key);

    return value;
}

}

namespace Transforms {

template<typename T>
class ITransform{
    public:
        ITransform() = default;
        virtual T run(const T& inp) = 0;
        // some transform needs its params to be updated (at runtime usually).
        bool aSetScale = false;
        bool aSetSize = false;
};

// image transforms
class ConvertColorImg: public ITransform<cv::cuda::GpuMat>{
    public:
        ConvertColorImg(const ColorMode colorMode = ColorMode::RGB): aColorMode(colorMode){};
        ConvertColorImg(const cv::FileNode& ocvfn){
            this->aColorMode = mapFileNodeString<ColorMode>(ocvfn["model"]);
        }
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp) override;
    private:
        ColorMode aColorMode;
};

class ResizeImg: public ITransform<cv::cuda::GpuMat>{
    public:
        ResizeImg(
            const cv::Size size = {-1, -1},
            const ResizeMethod method = ResizeMethod::scale): aSize(size), aMethod(method){};
        ResizeImg(const cv::FileNode& ocvfn){
            this->aSize = cv::Point2i(Utils::parseFileNodeCVVec<int, 2>(ocvfn["size"]));
            this->aMethod = mapFileNodeString<ResizeMethod>(ocvfn["method"]);
            if(this->aSize == cv::Size(Utils::MAX_INT, Utils::MAX_INT)){
                this->aSetSize = true;
            }
        }
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp) override;
        // Used by detector to define size (from net) at runtime
        void mSetSize(const cv::Size size){ this->aSize = size; };
    private:
        cv::Size aSize;
        ResizeMethod aMethod; // defines resizing method ["maintain_ar" or "scale"]
};

class CastImg: public ITransform<cv::cuda::GpuMat>{
    public:
        CastImg(const Precision dType = Precision::FP32, bool scale = true): aDType(dType), aScale(scale){};
        CastImg(const cv::FileNode& ocvfn){
            this->aDType = mapFileNodeString<Precision>(ocvfn["dtype"]);
            this->aScale = Utils::parseFileNodeValue<bool>(ocvfn["scale"]);
        }
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp) override;
    private:
        Precision aDType = Precision::FP32;
        bool aScale = true;
};

class NormalizeImg: public ITransform<cv::cuda::GpuMat>{
    public:
        NormalizeImg(
            const cv::Vec3f mean = {0.485, 0.456, 0.406},
            const cv::Vec3f std = {0.229, 0.224, 0.225}): aMean(mean), aStd(std){};
        NormalizeImg(const cv::FileNode& ocvfn){
            this->aMean = Utils::parseFileNodeCVVec<float, 3>(ocvfn["mean"]);
            this->aStd = Utils::parseFileNodeCVVec<float, 3>(ocvfn["std"]);
        }
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp) override;
    private:
        cv::Vec3f aMean;
        cv::Vec3f aStd;
        // Carefull with normalization parameters:
        // https://discuss.pytorch.org/t/discussion-why-normalise-according
        // -to-imagenet-mean-and-std-dev-for-transfer-learning/115670/7
};

// bboxes transforms
class ConvertBBox: public ITransform<BoundingBox>{
    public:
        ConvertBBox(const BoxFormat format = BoxFormat::xywh): aFormat(format){};
        ConvertBBox(const cv::FileNode& ocvfn){
            this->aFormat = mapFileNodeString<BoxFormat>(ocvfn["format"]);
        }
        BoundingBox run(const BoundingBox& inp) override;
    private:
        BoxFormat aFormat;
};

class RescaleBBox: public ITransform<BoundingBox>{
    public:
        RescaleBBox(
            const cv::Vec2f offset = {0.f, 0.f},
            const cv::Vec2f scale = {1.f, 1.f}): aOffset(offset), aScale(scale){};
        RescaleBBox(const cv::FileNode& ocvfn){
            this->aOffset = Utils::parseFileNodeCVVec<float, 2>(ocvfn["offset"]);
            this->aScale = Utils::parseFileNodeCVVec<float, 2>(ocvfn["scale"]);
            if(this->aScale == cv::Vec2f(Utils::MAX_FLOAT, Utils::MAX_FLOAT)){
                this->aSetScale = true;
            }
        }
        BoundingBox run(const BoundingBox& inp) override;
        // Used by detector to define scale (from net) at runtime
        void mSetScale(const cv::Vec2f scale){ this->aScale = scale; };
    private:
        cv::Vec2f aOffset;
        cv::Vec2f aScale;
};

class ResizeBBox: public ITransform<BoundingBox>{
    public:
        ResizeBBox(
            const cv::Size size = {-1, -1},
            const ResizeMethod method = ResizeMethod::scale): aSize(size), aMethod(method){};
        ResizeBBox(const cv::FileNode& ocvfn){
            this->aSize = cv::Point2i(Utils::parseFileNodeCVVec<int, 2>(ocvfn["size"]));
            this->aMethod = mapFileNodeString<ResizeMethod>(ocvfn["method"]);
            if(this->aSize == cv::Size(Utils::MAX_INT, Utils::MAX_INT)){
                this->aSetSize = true;
            }
        }
        BoundingBox run(const BoundingBox& inp) override;
        // Used by detector to define size (from image) at runtime
        void mSetSize(const cv::Size size){ this->aSize = size; };
    private:
        cv::Size aSize;
        ResizeMethod aMethod; // defines resizing method ["maintain_ar" or "scale"]
};

class FilterBBoxes: public ITransform<std::vector<BoundingBox>>{
    public:
        FilterBBoxes(float thresh = 0.1): aThresh(thresh){};
        FilterBBoxes(const cv::FileNode& ocvfn){
            this->aThresh = Utils::parseFileNodeValue<float>(ocvfn["thresh"]);
        }
        std::vector<BoundingBox> run(const std::vector<BoundingBox>& inp) override;
    private:
        float aThresh;
};

class NMSBBoxes: public ITransform<std::vector<BoundingBox>>{
    public:
        NMSBBoxes(
            float maxOverlap = 0.50,
            float nmsScaleFactor = 1.0,
            float outputScaleFactor = 1.0):
                aMaxOverlap(maxOverlap), aNMSScaleFactor(nmsScaleFactor), aOutputScaleFactor(outputScaleFactor){};
        NMSBBoxes(const cv::FileNode& ocvfn){
            this->aMaxOverlap = Utils::parseFileNodeValue<float>(ocvfn["max_overlap"]);
            this->aNMSScaleFactor = Utils::parseFileNodeValue<float>(ocvfn["nms_scale_factor"]);
            this->aOutputScaleFactor = Utils::parseFileNodeValue<float>(ocvfn["output_scale_factor"]);
        }
        std::vector<BoundingBox> run(const std::vector<BoundingBox>& inp) override;
    private:
        float aMaxOverlap;
        float aNMSScaleFactor;
        float aOutputScaleFactor;
};

}
