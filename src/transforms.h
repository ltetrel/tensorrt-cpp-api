#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "engine.h"
#include "boundingBox.h"


enum class ResizeMethod {
    maintain_ar,
    scale,
};

enum class ColorModel{
    // can be any of https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
    RGB,
    BGR,
    GRAY,
};

namespace Transforms {

template<typename T>
class ITransform{
    public:
        ITransform() = default;
        virtual T run(const T& inp) = 0;
};

// image transforms
class ResizeImg: public ITransform<cv::cuda::GpuMat>{
    public:
        ResizeImg(const cv::Size size = {-1, -1}, const ResizeMethod method = ResizeMethod::scale): aSize(size), aMethod(method){};
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp);
        void mSetSize(const cv::Size size){ this->aSize = size; }; // Used by detector to define size (from net) at runtime
    private:
        cv::Size aSize;
        ResizeMethod aMethod; // defines resizing method ["maintain_ar" or "scale"]
};

class ConvertColorImg: public ITransform<cv::cuda::GpuMat>{
    public:
        ConvertColorImg(const ColorModel colorModel = ColorModel::RGB): aColorModel(colorModel){};
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp);
    private:
        ColorModel aColorModel;
};

class CastImg: public ITransform<cv::cuda::GpuMat>{
    public:
        CastImg(const Precision dType = Precision::FP32, bool scale = true): aDType(dType), aScale(scale){};
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp);
    private:
        Precision aDType = Precision::FP32;
        bool aScale = true;
};

class NormalizeImg: public ITransform<cv::cuda::GpuMat>{
    public:
        NormalizeImg(
            const cv::Vec3f mean = {0.485, 0.456, 0.406},
            const cv::Vec3f std = {0.229, 0.224, 0.225}): aMean(mean), aStd(std){};
        cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp);
    private:
        cv::Vec3f aMean;
        cv::Vec3f aStd;
        // Carefull with normalization parameters:
        // https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670/7
};

// bboxes transforms
class ConvertBBox: public ITransform<BoundingBox>{
    public:
        ConvertBBox(const BoxFormat format = BoxFormat::xywh): aFormat(format){};
        BoundingBox run(const BoundingBox& inp) override;
    private:
        BoxFormat aFormat;
};

class RescaleBBox: public ITransform<BoundingBox>{
    public:
        RescaleBBox(const cv::Vec2f offset = {0.f, 0.f}, const cv::Vec2f scale = {1.f, 1.f}): aOffset(offset), aScale(scale){};
        BoundingBox run(const BoundingBox& inp) override;
        void mSetScale(const cv::Vec2f scale){ this->aScale = scale; }; // Used by detector to define scale (from net) at runtime
    private:
        cv::Vec2f aOffset;
        cv::Vec2f aScale;
};

class ResizeBBox: public ITransform<BoundingBox>{
    public:
        ResizeBBox(const cv::Size size = {-1, -1}, const ResizeMethod method = ResizeMethod::scale): aSize(size), aMethod(method){};
        BoundingBox run(const BoundingBox& inp) override;
        void mSetSize(const cv::Size size){ this->aSize = size; }; // Used by detector to define size (from image) at runtime
    private:
        cv::Size aSize;
        ResizeMethod aMethod; // defines resizing method ["maintain_ar" or "scale"]
};

class FilterBBoxes: public ITransform<std::vector<BoundingBox>>{
    public:
        FilterBBoxes(float thresh = 0.1): aThresh(thresh){};
        std::vector<BoundingBox> run(const std::vector<BoundingBox>& inp) override;
    private:
        float aThresh;
};

class NMSBBoxes: public ITransform<std::vector<BoundingBox>>{
    public:
        NMSBBoxes(float maxOverlap = 0.50, float nmsScaleFactor = 1.0, float outputScaleFactor = 1.0):
            aMaxOverlap(maxOverlap), aNMSScaleFactor(nmsScaleFactor), aOutputScaleFactor(outputScaleFactor){};
        std::vector<BoundingBox> run(const std::vector<BoundingBox>& inp) override;
    private:
        float aMaxOverlap;
        float aNMSScaleFactor;
        float aOutputScaleFactor;
};

}
