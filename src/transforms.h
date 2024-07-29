#pragma once

#include "engine.h"


enum class ResizeMethod {
    maintain_ar,
    scale,
};

enum class BoxFormat{
// https://pytorch.org/vision/main/_modules/torchvision/tv_tensors/_bounding_boxes.html#BoundingBoxFormat
    xywh,
    cxcywh,
    xyxy,
};

enum class ColorModel{
    // can be any of https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
    RGB,
    BGR,
    GRAY,
};

struct ResizeImg{
        // original height and width of the network
        cv::Size tgtSize;
        ResizeMethod method = ResizeMethod::scale; // defines resizing method ["maintain_ar" or "scale"]
};

struct ConvertColor{
    ColorModel model = ColorModel::RGB;
};

struct CastImg{
    Precision dtype = Precision::FP32;
    bool scale = true;
};

struct NormalizeImg{
    // Carefull with normalization parameters:
    // https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670/7
    cv::Vec3f mean = {0.485, 0.456, 0.406};
    cv::Vec3f std = {0.229, 0.224, 0.225};
};

struct FilterBoxes{
    float thresh = 0.1;  // probability for an object to exists (yolo objectness)
};

struct ConvertBox{
    BoxFormat srcFmt = BoxFormat::xyxy;  // can be either "xyxy", "cxcywh" or "xywh"
};

struct RescaleBox{
    cv::Vec2f offset = {0.f, 0.f};
    cv::Vec2f scale = {1.f, 1.f};
};

struct ResizeBox{
    cv::Size inpSize;
    cv::Size tgtSize;
    ResizeMethod method = ResizeMethod::scale;
};

struct NMS{
    float maxOverlap = 0.50;
    float nmsScaleFactor = 1.0;
    float outputScaleFactor = 1.0;
};

namespace Transforms {
    //TODO: create class for transform functions
    class ResizeImg{
        public:
            ResizeImg(cv::Size tgtSize, ResizeMethod method): aTgtSize(tgtSize), aMethod(method){};
            //TODO: constructor from a cv::FileNode
            cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp);
        private:
            cv::Size aTgtSize;
            ResizeMethod aMethod = ResizeMethod::scale; // defines resizing method ["maintain_ar" or "scale"]
    };
// image transforms
cv::cuda::GpuMat resizeImg(const cv::cuda::GpuMat& inp, const cv::Size size, const ResizeMethod method);
cv::cuda::GpuMat convertColorImg(const cv::cuda::GpuMat& inp, const ColorModel tgtModel);
cv::cuda::GpuMat castImg(
    const cv::cuda::GpuMat& inp,
    const Precision dType = Precision::FP32,
    const bool scale = true);
cv::cuda::GpuMat normalizeImg(const cv::cuda::GpuMat& inp, const cv::Scalar mean, const cv::Scalar std);
// bboxes transforms
std::vector<unsigned int> getValidBoxIds(const std::vector<float>& inp, float thresh);
cv::Vec4f convertBox(const cv::Vec4f& inp, const BoxFormat srcFormat, const BoxFormat tgtFormat = BoxFormat::xywh);
cv::Vec4f rescaleBox(const cv::Vec4f& inp, const cv::Vec2f offset, const cv::Vec2f scale);
cv::Vec4f resizeBox(const cv::Vec4f& inp, const cv::Size inpCanvaSize, const cv::Size tgtCanvaSize, const ResizeMethod method);
// cv::dnn::NMSBoxesBatched not needed
}
