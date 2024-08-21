#include <type_traits>

#include "configParser.h"

// #include "utils.h"

namespace {
    std::unordered_map<std::string, ModelType> const mapModelType = {
        {"darknet", ModelType::darknet},
        {"netharn", ModelType::netharn}
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

TargetPostTransforms CfgParser::mParsePostProcessing(cv::FileStorage inputFs){
    // Currently only support FilterBoxes, ConvertBox, RescaleBBox, ResizeBox and NMS
    //TODO: in near future, will dynamically read from yaml
    TargetPostTransforms targetPostTransforms;
    cv::FileNode ocvfnPreprocess = inputFs["target_post_transforms"];

    // FilterBoxes
    targetPostTransforms.filter = Transforms::FilterBBoxes(ocvfnPreprocess["FilterBoxes"]);
    // ConvertBox
    targetPostTransforms.convert = Transforms::ConvertBBox(ocvfnPreprocess["ConvertBox"]);
    // RescaleBBox
    targetPostTransforms.rescale = Transforms::RescaleBBox(ocvfnPreprocess["RescaleBBox"]);
    // ResizeBBox
    targetPostTransforms.resize = Transforms::ResizeBBox(ocvfnPreprocess["ResizeBox"]);

    // NMS
    targetPostTransforms.nms = Transforms::NMSBBoxes(ocvfnPreprocess["NMS"]);

    return targetPostTransforms;
}

BoxFormat CfgParser::mParseBBoxFormat(cv::FileStorage inputFs){
    BoxFormat srcFormat;
    
    cv::FileNode ocvfn = inputFs["target_post_transforms"]["ConvertBox"]["src_fmt"];
    srcFormat = mapFileNodeString<BoxFormat>(ocvfn);

    return srcFormat;
}

Model CfgParser::mParseModel(cv::FileStorage inputFs){
    Model model;

    cv::FileNode ocvfnModel = inputFs["model"];
    ModelType modelType = Utils::getValueFromMapKey<ModelType>(mapModelType, ocvfnModel["type"]);
    model = Model{modelType};

    return model;
}

std::vector<std::string> CfgParser::mParseLabels(cv::FileStorage inputFs){
    std::vector<std::string> labels;
    cv::FileNode ocvfnLabels = inputFs["labels"];

    labels = Utils::parseFileNodeVector<std::string>(ocvfnLabels);

    return labels;
}

std::vector<std::vector<float>> CfgParser::mParseColors(cv::FileStorage inputFs){
    std::vector<std::vector<float>> colors;
    cv::FileNode ocvfnColors = inputFs["colors"];

    for (cv::FileNodeIterator it = ocvfnColors.begin(); it != ocvfnColors.end(); ++it){
        std::vector<float> color;
        color = Utils::parseFileNodeVector<float>(*it);
        colors.emplace_back(color);
    }

    return colors;
}

CfgParser::CfgParser(std::filesystem::path cfgPath){
    cv::FileStorage fs(cfgPath, cv::FileStorage::READ);

    this->aImagePreTransforms = this->mParsePreProcessing(fs);
    this->aTargetPostTransforms = this->mParsePostProcessing(fs);
    this->aBBoxSrcFormat = this->mParseBBoxFormat(fs);
    this->aModel = this->mParseModel(fs);
    this->aLabels = this->mParseLabels(fs);
    this->aColors = this->mParseColors(fs);

    fs.release();
}
