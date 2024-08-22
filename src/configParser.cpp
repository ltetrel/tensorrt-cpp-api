#include <type_traits>

#include "configParser.h"

#include "utils.h"


namespace {

std::unordered_map<std::string, ModelType> const mapModelType = {
    {"yolov4-csp-s-mish", ModelType::yolov4},
    {"netharn", ModelType::cascadeRCNN}
};

std::unordered_map<std::string, ModelBackend> const mapModelBackend = {
    {"darknet", ModelBackend::darknet},
    {"netharn", ModelBackend::netharn}
};

}

std::vector<ITImgTransformPtr> CfgParser::mParsePreProcessing(cv::FileStorage inputFs){
    std::vector<ITImgTransformPtr> transforms;

    cv::FileNode ocvfnPreprocess = inputFs["image_pre_transforms"];
    for (cv::FileNodeIterator it = ocvfnPreprocess.begin(); it != ocvfnPreprocess.end(); it++){
        cv::FileNode ocvfn = *it;
        std::string ocvfnName = ocvfn.name();
        if (ocvfnName == "ConvertColorImg"){
            transforms.emplace_back(
                std::make_shared<Transforms::ConvertColorImg>(Transforms::ConvertColorImg(ocvfn)));
        }
        else if (ocvfnName == "ResizeImg"){
            transforms.emplace_back(
                std::make_shared<Transforms::ResizeImg>(Transforms::ResizeImg(ocvfn)));
        }
        else if (ocvfnName == "CastImg"){
            transforms.emplace_back(
                std::make_shared<Transforms::CastImg>(Transforms::CastImg(ocvfn)));
        }
        else if (ocvfnName == "NormalizeImg"){
            transforms.emplace_back(
                std::make_shared<Transforms::NormalizeImg>(Transforms::NormalizeImg(ocvfn)));
        }
        else{
            throw std::runtime_error("Transform \"" + ocvfnName + "\" is not implemented!");
        }
    }

    return transforms;
}

std::vector<ITBBoxTransformPtr> CfgParser::mParsePostProcessing(cv::FileStorage inputFs){
    std::vector<ITBBoxTransformPtr> transforms;

    cv::FileNode ocvfnPostprocess = inputFs["target_post_transforms"];
    for (cv::FileNodeIterator it = ocvfnPostprocess.begin(); it != ocvfnPostprocess.end(); it++){
        cv::FileNode ocvfn = *it;
        std::string ocvfnName = ocvfn.name();
        if (ocvfnName == "FilterBBoxes"){
            this->aTargetsFilterTransform = 
                std::make_shared<Transforms::FilterBBoxes>(Transforms::FilterBBoxes(ocvfn));
        }
        else if (ocvfnName == "ConvertBBox"){
            transforms.emplace_back(
                std::make_shared<Transforms::ConvertBBox>(Transforms::ConvertBBox(ocvfn)));
        }
        else if (ocvfnName == "RescaleBBox"){
            transforms.emplace_back(
                std::make_shared<Transforms::RescaleBBox>(Transforms::RescaleBBox(ocvfn)));
        }
        else if (ocvfnName == "ResizeBBox"){
            transforms.emplace_back(
                std::make_shared<Transforms::ResizeBBox>(Transforms::ResizeBBox(ocvfn)));
        }
        else if (ocvfnName == "NMSBBoxes"){
            this->aTargetsNMSTransform =
                std::make_shared<Transforms::NMSBBoxes>(Transforms::NMSBBoxes(ocvfn));
        }
        else{
            throw std::runtime_error("Transform \"" + ocvfnName + "\" is not implemented!");
        }
    }

    return transforms;
}

Model CfgParser::mParseModel(cv::FileStorage inputFs){
    Model model;

    cv::FileNode ocvfnModel = inputFs["model"];
    ModelType modelType = Utils::getValueFromMapKey<ModelType>(mapModelType, ocvfnModel["type"]);
    ModelBackend modelBackend = Utils::getValueFromMapKey<ModelBackend>(mapModelBackend, ocvfnModel["backend"]);
    model = Model{modelType, modelBackend};

    return model;
}

std::vector<std::string> CfgParser::mParseLabels(cv::FileStorage inputFs){
    std::vector<std::string> labels;
    cv::FileNode ocvfnLabels = inputFs["labels"];

    labels = Utils::parseFileNodeVector<std::string>(ocvfnLabels);

    return labels;
}

std::vector<cv::Scalar> CfgParser::mParseColors(cv::FileStorage inputFs){
    std::vector<cv::Scalar> colors;
    cv::FileNode ocvfnColors = inputFs["colors"];

    for (cv::FileNodeIterator it = ocvfnColors.begin(); it != ocvfnColors.end(); ++it){
        cv::Scalar color;
        color = Utils::parseFileNodeCVVec<float, 3>(*it);
        colors.emplace_back(color);
    }

    return colors;
}

CfgParser::CfgParser(std::filesystem::path cfgPath){
    cv::FileStorage fs(cfgPath, cv::FileStorage::READ);

    this->aImagePreTransforms = this->mParsePreProcessing(fs);
    this->aTargetPostTransforms = this->mParsePostProcessing(fs);
    this->aModel = this->mParseModel(fs);
    this->aLabels = this->mParseLabels(fs);
    this->aColors = this->mParseColors(fs);

    fs.release();
}
