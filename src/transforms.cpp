#include "transforms.h"
#include <opencv2/cudaimgproc.hpp>


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

cv::cuda::GpuMat Transforms::toDtypeImg(const cv::cuda::GpuMat& inp, const Precision dType, const bool scale){
    cv::cuda::GpuMat mat;

    return mat;
    // if (normalize) {
    //     // [0.f, 1.f]
    //     gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    // } else {
    //     // [0.f, 255.f]
    //     gpu_dst.convertTo(mfloat, CV_32FC3);
    // }
}

cv::cuda::GpuMat Transforms::normalizeImg(const cv::cuda::GpuMat& inp, const cv::Scalar mean, const cv::Scalar std){
    cv::cuda::GpuMat mat;

    return mat;
    // cv:Mat mfloat;

    // cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
    // cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, cv::noArray(), -1);
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
