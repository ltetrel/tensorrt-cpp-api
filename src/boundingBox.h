#pragma once

#include <opencv2/opencv.hpp>

enum class BoxFormat{
// https://pytorch.org/vision/main/_modules/torchvision/tv_tensors/_bounding_boxes.html#BoundingBoxFormat
    xywh,
    cxcywh,
    xyxy,
};

// same as torchvision.datapoints.BoundingBox
class BoundingBox{
public:
    BoundingBox() = default;
    BoundingBox(const BoundingBox&) = default;
        BoundingBox(const cv::Vec4f& bounds, const cv::Size& size, const BoxFormat& format, const int label, const float conf):
        aBounds(bounds), aSize(size), aBoxFormat(format), aLabel(label), aConf(conf){}
    BoundingBox(const cv::Vec4f& bounds, const cv::Mat& img, const BoxFormat& format, const int label, const float conf):
        aBounds(bounds), aSize(img.size()), aBoxFormat(format), aLabel(label), aConf(conf){}

    cv::Vec4f aBounds;
    cv::Size aSize;
    BoxFormat aBoxFormat;
    int aLabel;
    float aConf;
};
