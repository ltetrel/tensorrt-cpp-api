#include "transforms.h"
#include <opencv2/cudaimgproc.hpp>



cv::cuda::GpuMat Transforms::ResizeImg::run(const cv::cuda::GpuMat& inp){
    cv::cuda::GpuMat resized(this->aTgtSize, inp.type());
    cv::cuda::GpuMat scaled;

    double rx = static_cast<double>(this->aTgtSize.width)/static_cast<double>(inp.cols);
    double ry = static_cast<double>(this->aTgtSize.height)/static_cast<double>(inp.rows);
    if (this->aMethod == ResizeMethod::maintain_ar){
        rx = std::min(rx, ry);
        ry = rx;
    }

    cv::cuda::resize(inp, scaled, cv::Size(), rx, ry);
    // copy rescaled input to roi, needed for maintain_ar
    cv::cuda::GpuMat roi(resized, cv::Rect(0, 0, scaled.cols, scaled.rows));          
    scaled.copyTo(roi);

    return resized;
}

cv::cuda::GpuMat Transforms::resizeImg(const cv::cuda::GpuMat& inp, const cv::Size size, const ResizeMethod method){
    cv::cuda::GpuMat resized(size, inp.type());
    cv::cuda::GpuMat scaled;

    double rx = static_cast<double>(size.width)/static_cast<double>(inp.cols);
    double ry = static_cast<double>(size.height)/static_cast<double>(inp.rows);
    if (method == ResizeMethod::maintain_ar){
        rx = std::min(rx, ry);
        ry = rx;
    }

    cv::cuda::resize(inp, scaled, cv::Size(), rx, ry);
    // copy rescaled input to roi, needed for maintain_ar
    cv::cuda::GpuMat roi(resized, cv::Rect(0, 0, scaled.cols, scaled.rows));          
    scaled.copyTo(roi);

    return resized;
}

cv::cuda::GpuMat Transforms::convertColorImg(const cv::cuda::GpuMat& inp, const ColorModel tgtModel){
    cv::cuda::GpuMat colored;

    // Assumes an input in BGR opencv format
    switch (tgtModel)
    {
    case ColorModel::BGR:
        colored = inp;
        break;
    case ColorModel::RGB:
        cv::cuda::cvtColor(inp, colored, cv::COLOR_BGR2RGB);
        break;
    case ColorModel::GRAY:
        cv::cuda::cvtColor(inp, colored, cv::COLOR_BGR2GRAY);
        break;

    default:
        colored = inp;
    }

    return colored;
}

cv::cuda::GpuMat Transforms::castImg(const cv::cuda::GpuMat& inp, const Precision dType, const bool scale){
    cv::cuda::GpuMat converted;
    double alpha = 1.0;

    if (dType != Precision::FP32){
        throw std::runtime_error("Will cast only to FP32 as TensorRT works best with that.\n"
            "For more information see: "
            "https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861"
            "/developer-guide/index.html#reformat-free-network-tensors");
    }

    if (scale){
        switch (inp.depth()){
            case CV_8U:
                alpha = 1. / static_cast<double>(std::numeric_limits<unsigned char>::max());
                break;
            case CV_8S:
                alpha = 1. / static_cast<double>(std::numeric_limits<char>::max());
                break;
            case CV_16U:
                alpha = 1. / static_cast<double>(std::numeric_limits<unsigned short>::max());
                break;
            case CV_16S:
                alpha = 1. / static_cast<double>(std::numeric_limits<short>::max());
                break;
            case CV_32S:
                alpha = 1. / static_cast<double>(std::numeric_limits<int>::max());
                break;
            case CV_32F:
                alpha = 1. / static_cast<double>(std::numeric_limits<float>::max());
                break;
            case CV_64F:
                alpha = 1. / std::numeric_limits<double>::max();
                break;
            default:
                alpha = 1.0;
                break;
        }
    }

    inp.convertTo(converted, CV_32FC3, alpha);

    return converted;
}

cv::cuda::GpuMat Transforms::normalizeImg(const cv::cuda::GpuMat& inp, const cv::Scalar mean, const cv::Scalar std){
    cv::cuda::GpuMat normalized;

    // Apply scaling and mean subtraction
    cv::cuda::subtract(inp, mean, normalized, cv::noArray(), -1);
    cv::cuda::divide(normalized, std, normalized, 1, -1);

    return normalized;
}

std::vector<unsigned int> Transforms::getValidBoxIds(std::vector<float>& inp, float thresh){
    std::vector<unsigned int> validBoxIds;

    for (size_t i=0; i < inp.size(); i++){
        if(inp[i] > thresh){
            validBoxIds.push_back(i);
        }
    }

    return validBoxIds;
}

cv::Vec4f Transforms::convertBox(const cv::Vec4f& inp, const BoxFormat srcFormat, const BoxFormat tgtFormat){
    cv::Vec4f converted;

    switch (srcFormat)
    {
    case BoxFormat::cxcywh:
        converted[0] = inp[0] - (inp[2] / 2.f);
        converted[1] = inp[1] - (inp[3] / 2.f);
        converted[2] = inp[2];
        converted[3] = inp[3];
        break;
    case BoxFormat::xyxy:
        converted[0] = inp[0];
        converted[1] = inp[1];
        converted[2] = inp[2] - inp[0];
        converted[3] = inp[3] - inp[1];
        break;
    
    default:
        converted = inp;
    }

    return converted;
}

cv::Vec4f Transforms::rescaleBox(const cv::Vec4f& inp, const cv::Vec2f offset, const cv::Vec2f scale){
    cv::Vec4f rescaled;

    rescaled[0] = scale[0]*(inp[0] + offset[0]);
    rescaled[1] = scale[1]*(inp[1] + offset[1]);
    rescaled[2] = scale[0]*(inp[2] + offset[0]);
    rescaled[3] = scale[1]*(inp[3] + offset[1]);

    return rescaled;
}

cv::Vec4f Transforms::resizeBox(const cv::Vec4f& inp, const cv::Size inpCanvaSize, const cv::Size tgtCanvaSize, const ResizeMethod method){
    cv::Vec4f resized;

    float rx = static_cast<float>(tgtCanvaSize.width)/static_cast<float>(inpCanvaSize.width);
    float ry = static_cast<float>(tgtCanvaSize.height)/static_cast<float>(inpCanvaSize.height);
    if (method == ResizeMethod::maintain_ar){
        rx = std::max(rx, ry);  //std::max to make sure to cover all target canva region
        ry = rx;
    }
    resized[0] = inp[0] * rx;
    resized[1] = inp[1] * ry;
    resized[2] = inp[2] * rx;
    resized[3] = inp[3] * ry;

    return resized;
}
