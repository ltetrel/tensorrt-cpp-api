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

struct ResizeImg{
        cv::Size size;  // original height and width of the network
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
    cv::Size size;  // normally extracted from input frame
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
            ResizeImg(cv::Size size, ResizeMethod method): aSize(size), aMethod(method){};
            //TODO: constructor from a cv::FileNode
            cv::cuda::GpuMat run(const cv::cuda::GpuMat& inp);
        private:
            cv::Size aSize;
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
BoundingBox convertBBox(const BoundingBox& inp, const BoxFormat format = BoxFormat::xywh);
BoundingBox rescaleBBox(const BoundingBox& inp, const cv::Vec2f offset, const cv::Vec2f scale);
BoundingBox resizeBBox(const BoundingBox& inp, const cv::Size size, const ResizeMethod method);
std::vector<BoundingBox> nmsBBox(
    const std::vector<BoundingBox>& inp, const float maxOverlap, const float nmsScaleFactor, const float outputScaleFactor);
}
