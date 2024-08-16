#include <type_traits>

#include "configParser.h"


namespace {  
    template <typename T>
    T parseFileNodeValue(cv::FileNode ocvFn){
        T value;
        
        if constexpr(std::is_same<T, bool>::value){
            value = (ocvFn.string() == "true") ? true : false;
        }
        else{
            value = ocvFn;
        }

        return value;
    }

    template <typename T>
    std::vector<T> parseFileNodeVector(cv::FileNode ocvFn){
        std::vector<T> values;

        for (cv::FileNodeIterator it = ocvFn.begin(); it != ocvFn.end(); ++it){
            values.emplace_back(static_cast<T>(*it));
        }

        return values;
    }

    struct TransformValueMapper{
        std::unordered_map<std::string, ColorModel> const colorModel = {
            {"RGB", ColorModel::RGB},
            {"BGR", ColorModel::BGR},
            {"GRAY", ColorModel::GRAY}
        };
        std::unordered_map<std::string, ResizeMethod> const resizeMethod = {
            {"maintain_ar", ResizeMethod::maintain_ar},
            {"scale", ResizeMethod::scale}
        };
        std::unordered_map<std::string, Precision> const imageType = {
            {"int", Precision::INT8},
            {"float", Precision::FP32}
        };
        std::unordered_map<std::string, BoxFormat> const boxFormat = {
            {"xyxy", BoxFormat::xyxy},
            {"cxcywh", BoxFormat::cxcywh},
            {"xywh", BoxFormat::xywh}
        };
        std::unordered_map<std::string, ModelType> const modelType = {
            {"darknet", ModelType::darknet},
            {"netharn", ModelType::netharn}
        };
    };

    template <typename T>
    T mapFileNodeString(cv::FileNode ocvFn){
        std::string key = ocvFn.string();
        T value;
        TransformValueMapper transformValueMapper;
        std::unordered_map<std::string, T> mapperTable;

        if constexpr(std::is_same<T, ColorModel>::value){
            mapperTable = transformValueMapper.colorModel;
        }
        else if constexpr(std::is_same<T, ResizeMethod>::value){
            mapperTable = transformValueMapper.resizeMethod;
        }
        else if constexpr(std::is_same<T, Precision>::value){
            mapperTable = transformValueMapper.imageType;
        }
        else if constexpr(std::is_same<T, BoxFormat>::value){
            mapperTable = transformValueMapper.boxFormat;
        }
        else if constexpr(std::is_same<T, ModelType>::value){
            mapperTable = transformValueMapper.modelType;
        }

        auto it = mapperTable.find(key);
        if (it != mapperTable.end()) {
            value = it->second;
        }
        else{
            throw std::runtime_error("Cannot find key: " + key);
        }

        return value;
    }
}

ImagePreTransforms CfgParser::mParsePreProcessing(cv::FileStorage inputFs){
    // Currently only support ConvertColor, ResizeImg, CastImg and NormalizeImg
    //TODO: in near future, will dynamically read from yaml
    ImagePreTransforms imagePreTransforms;
    cv::FileNode ocvfnPreprocess = inputFs["image_pre_transforms"];

    // ConvertColor
    cv::FileNode ocvfnConvertColor = ocvfnPreprocess["ConvertColorImg"];
    ColorModel colorModel = mapFileNodeString<ColorModel>(ocvfnConvertColor["model"]);
    imagePreTransforms.convertColor = Transforms::ConvertColorImg(colorModel);
    // ResizeImg
    cv::FileNode ocvfnResizeImg = ocvfnPreprocess["ResizeImg"];
    std::vector<int> size = parseFileNodeVector<int>(ocvfnResizeImg["size"]);
    ResizeMethod resizeMethod = mapFileNodeString<ResizeMethod>(ocvfnResizeImg["method"]);
    imagePreTransforms.resize = Transforms::ResizeImg(cv::Size(size[0], size[1]), resizeMethod);
    // CastImg
    cv::FileNode ocvfnCastImg = ocvfnPreprocess["CastImg"];
    Precision dtype = mapFileNodeString<Precision>(ocvfnCastImg["dtype"]);
    bool scale = parseFileNodeValue<bool>(ocvfnCastImg["scale"]);
    imagePreTransforms.cast = Transforms::CastImg(dtype, scale);
    // NormalizeImg
    cv::FileNode ocvfnNormImg = ocvfnPreprocess["NormalizeImg"];
    std::vector<float> means = parseFileNodeVector<float>(ocvfnNormImg["mean"]);
    std::vector<float> stds = parseFileNodeVector<float>(ocvfnNormImg["std"]);
    imagePreTransforms.normalize = Transforms::NormalizeImg{
        cv::Vec3f(means[0], means[1], means[2]),
        cv::Vec3f(stds[0], stds[1], stds[2])
    };

    return imagePreTransforms;
}

TargetPostTransforms CfgParser::mParsePostProcessing(cv::FileStorage inputFs){
    // Currently only support FilterBoxes, ConvertBox, RescaleBox, ResizeBox and NMS
    //TODO: in near future, will dynamically read from yaml
    TargetPostTransforms targetPostTransforms;
    cv::FileNode ocvfnPreprocess = inputFs["target_post_transforms"];

    // FilterBoxes
    cv::FileNode ocvfnFilterBoxes = ocvfnPreprocess["FilterBoxes"];
    float thresh = parseFileNodeValue<float>(ocvfnFilterBoxes["thresh"]);
    targetPostTransforms.filter = Transforms::FilterBBoxes(thresh);
    // ConvertBox
    cv::FileNode ocvfnConvertBox = ocvfnPreprocess["ConvertBox"];
    BoxFormat srcFormat = mapFileNodeString<BoxFormat>(ocvfnConvertBox["src_fmt"]);
    this->aBBoxSrcFormat = srcFormat;
    BoxFormat tgtFormat = mapFileNodeString<BoxFormat>(ocvfnConvertBox["tgt_fmt"]);
    targetPostTransforms.convert = Transforms::ConvertBBox(tgtFormat);
    // RescaleBox
    cv::FileNode ocvfnRescaleBox = ocvfnPreprocess["RescaleBox"];
    std::vector<float> offset = parseFileNodeVector<float>(ocvfnRescaleBox["offset"]);
    std::vector<float> scale = parseFileNodeVector<float>(ocvfnRescaleBox["scale"]);
    targetPostTransforms.rescale = Transforms::RescaleBBox(
        cv::Vec2f(offset[0], offset[1]),
        cv::Vec2f(scale[0], scale[1])
    );
    // ResizeBBox
    cv::FileNode ocvfnResizeBox = ocvfnPreprocess["ResizeBox"];
    std::vector<int> size = parseFileNodeVector<int>(ocvfnResizeBox["size"]);
    ResizeMethod resizeMethod = mapFileNodeString<ResizeMethod>(ocvfnResizeBox["method"]);
    targetPostTransforms.resize = Transforms::ResizeBBox(cv::Size(size[0], size[1]), resizeMethod);

    // NMS
    cv::FileNode ocvfnNMS = ocvfnPreprocess["NMS"];
    float maxOverlap = parseFileNodeValue<float>(ocvfnNMS["max_overlap"]);
    float scaleFactor = parseFileNodeValue<float>(ocvfnNMS["nms_scale_factor"]);
    float outScaleFactor = parseFileNodeValue<float>(ocvfnNMS["output_scale_factor"]);
    targetPostTransforms.nms = Transforms::NMSBBoxes(maxOverlap, scaleFactor, outScaleFactor);

    return targetPostTransforms;
}

Model CfgParser::mParseModel(cv::FileStorage inputFs){
    Model model;

    cv::FileNode ocvfnModel = inputFs["model"];
    ModelType modelType = mapFileNodeString<ModelType>(ocvfnModel["type"]);
    model = Model{modelType};

    return model;
}

std::vector<std::string> CfgParser::mParseLabels(cv::FileStorage inputFs){
    std::vector<std::string> labels;
    cv::FileNode ocvfnLabels = inputFs["labels"];

    labels = parseFileNodeVector<std::string>(ocvfnLabels);

    return labels;
}

std::vector<std::vector<float>> CfgParser::mParseColors(cv::FileStorage inputFs){
    std::vector<std::vector<float>> colors;
    cv::FileNode ocvfnColors = inputFs["colors"];

    for (cv::FileNodeIterator it = ocvfnColors.begin(); it != ocvfnColors.end(); ++it){
        std::vector<float> color;
        color = parseFileNodeVector<float>(*it);
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
