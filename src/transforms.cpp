#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "transforms.h"


cv::cuda::GpuMat Transforms::ResizeImg::run(const cv::cuda::GpuMat& inp){
    cv::cuda::GpuMat resized(this->aSize, inp.type());
    cv::cuda::GpuMat scaled;

    double rx = static_cast<double>(this->aSize.width)/static_cast<double>(inp.cols);
    double ry = static_cast<double>(this->aSize.height)/static_cast<double>(inp.rows);
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

std::vector<unsigned int> Transforms::getValidBoxIds(const std::vector<float>& inp, float thresh){
    std::vector<unsigned int> validBoxIds;

    for (size_t i=0; i < inp.size(); i++){
        if(inp[i] > thresh){
            validBoxIds.push_back(i);
        }
    }

    return validBoxIds;
}

BoundingBox Transforms::convertBBox(const BoundingBox& inp, const BoxFormat format){
    //TODO: let user witch target format
    cv::Vec4f convertedBounds;

    switch (inp.aBoxFormat)
    {
    case BoxFormat::cxcywh:
        convertedBounds[0] = inp.aBounds[0] - (inp.aBounds[2] / 2.f);
        convertedBounds[1] = inp.aBounds[1] - (inp.aBounds[3] / 2.f);
        convertedBounds[2] = inp.aBounds[2];
        convertedBounds[3] = inp.aBounds[3];
        break;
    case BoxFormat::xyxy:
        convertedBounds[0] = inp.aBounds[0];
        convertedBounds[1] = inp.aBounds[1];
        convertedBounds[2] = inp.aBounds[2] - inp.aBounds[0];
        convertedBounds[3] = inp.aBounds[3] - inp.aBounds[1];
        break;
    
    default:
        convertedBounds = inp.aBounds;
    }

    BoundingBox convertedBBox(inp);
    convertedBBox.aBounds = convertedBounds;
    convertedBBox.aBoxFormat = BoxFormat::xywh;

    return convertedBBox;
}

BoundingBox Transforms::rescaleBBox(const BoundingBox& inp, const cv::Vec2f offset, const cv::Vec2f scale){
    cv::Vec4f rescaledBounds;

    rescaledBounds[0] = scale[0]*(inp.aBounds[0] + offset[0]);
    rescaledBounds[1] = scale[1]*(inp.aBounds[1] + offset[1]);
    rescaledBounds[2] = scale[0]*(inp.aBounds[2] + offset[0]);
    rescaledBounds[3] = scale[1]*(inp.aBounds[3] + offset[1]);

    BoundingBox rescaled(inp);
    rescaled.aBounds = rescaledBounds;

    return rescaled;
}

BoundingBox Transforms::resizeBBox(const BoundingBox& inp, const cv::Size size, const ResizeMethod method){
    cv::Vec4f resizedBounds;

    float rx = static_cast<float>(size.width) / static_cast<float>(inp.aSize.width);
    float ry = static_cast<float>(size.height) / static_cast<float>(inp.aSize.height);
    if (method == ResizeMethod::maintain_ar){
        rx = std::max(rx, ry);  //std::max to make sure to cover all target canva region
        ry = rx;
    }
    resizedBounds[0] = inp.aBounds[0] * rx;
    resizedBounds[1] = inp.aBounds[1] * ry;
    resizedBounds[2] = inp.aBounds[2] * rx;
    resizedBounds[3] = inp.aBounds[3] * ry;

    BoundingBox resized(inp);
    resized.aBounds = resizedBounds;

    return resized;
}

std::vector<BoundingBox> Transforms::nmsBBox(
        const std::vector<BoundingBox>& inp, const float maxOverlap, const float nmsScaleFactor, const float outputScaleFactor){
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> nmsIndices;

    // fill variables for opencv NMS
    for (auto& currBBox : inp){
        cv::Rect2f bbox;
        bbox.x = currBBox.aBounds[0];
        bbox.y = currBBox.aBounds[1];
        bbox.width = currBBox.aBounds[2];
        bbox.height = currBBox.aBounds[3];

        bboxes.push_back(bbox);
        labels.push_back(currBBox.aLabel);
        scores.push_back(currBBox.aConf);
    }
    // NMS batched version performs each class independently
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, 0.0, maxOverlap, nmsIndices);

    //return the list of bbox
    std::vector<BoundingBox> listBBoxNMSed;
    for (auto& currIdx : nmsIndices) {
        BoundingBox currBBox(inp[0]);  // the output nms-ed boxes will have same size and format as input
        currBBox.aConf = scores[currIdx];
        currBBox.aLabel = labels[currIdx];
        currBBox.aBounds = {
            static_cast<float>(bboxes[currIdx].x),
            static_cast<float>(bboxes[currIdx].y),
            static_cast<float>(bboxes[currIdx].width),
            static_cast<float>(bboxes[currIdx].height)
        };
        listBBoxNMSed.push_back(currBBox);
    }
    
    return listBBoxNMSed;
}
