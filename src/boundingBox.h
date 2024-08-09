#pragma once

#include <iostream>
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

    friend std::ostream& operator<<(std::ostream& out, const BoundingBox& other)
    {
        out << other.aLabel <<"\t" \
            << other.aBounds[0] <<"\t"<< other.aBounds[1] <<"\t"<< other.aBounds[2]<<"\t"<< other.aBounds[3] \
            <<"\t"<< std::fixed << std::setprecision(6) << other.aConf << std::setprecision(0);

        return out;
    }
    friend std::istream& operator>>(std::istream& in, BoundingBox& other)
    {
        in >> other.aLabel >> other.aBounds[0] >> other.aBounds[1] >> other.aBounds[2] >> other.aBounds[3] >> other.aConf;

        return in;
    }

    cv::Vec4f aBounds;
    cv::Size aSize;
    BoxFormat aBoxFormat;
    int aLabel;
    float aConf;
};
