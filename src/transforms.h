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

namespace Transforms {
// image transforms
cv::cuda::GpuMat resizeImg(const cv::cuda::GpuMat& inp, const cv::Size size, const ResizeMethod method);
cv::cuda::GpuMat convertColorImg(const cv::cuda::GpuMat& inp, const ColorModel tgtModel);
cv::cuda::GpuMat castImg(
    const cv::cuda::GpuMat& inp,
    const Precision dType = Precision::FP32,
    const bool scale = true);
cv::cuda::GpuMat normalizeImg(const cv::cuda::GpuMat& inp, const cv::Scalar mean, const cv::Scalar std);
// bboxes transforms
std::vector<unsigned int> getValidBoxIds(std::vector<float>& inp, float thresh);
cv::Vec4f convertBox(const cv::Vec4f& inp, const BoxFormat srcFormat, const BoxFormat tgtFormat = BoxFormat::xywh);
cv::Vec4f rescaleBox(const cv::Vec4f& inp, const cv::Vec2f offset, const cv::Vec2f scale);
cv::Vec4f resizeBox(const cv::Vec4f& inp, const cv::Size inpCanvaSize, const cv::Size tgtCanvaSize, const ResizeMethod method);
// cv::dnn::NMSBoxesBatched not needed
}
