#include <type_traits>

#include "configParser.h"


namespace{
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
    cv::FileNode ocvfnConvertColor = ocvfnPreprocess["ConvertColor"];
    ColorModel colorModel = mapFileNodeString<ColorModel>(ocvfnConvertColor["model"]);
    imagePreTransforms.convertColor = ConvertColor{colorModel};
    // ResizeImg
    cv::FileNode ocvfnResizeImg = ocvfnPreprocess["ResizeImg"];
    std::vector<int> size = parseFileNodeVector<int>(ocvfnResizeImg["size"]);
    ResizeMethod resizeMethod = mapFileNodeString<ResizeMethod>(ocvfnResizeImg["method"]);
    imagePreTransforms.resize = ResizeImg{{size[0], size[1]}, resizeMethod};
    // CastImg
    cv::FileNode ocvfnCastImg = ocvfnPreprocess["CastImg"];
    Precision dtype = mapFileNodeString<Precision>(ocvfnCastImg["dtype"]);
    bool scale = parseFileNodeValue<bool>(ocvfnCastImg["scale"]);
    imagePreTransforms.cast = CastImg{dtype, scale};
    // NormalizeImg
    cv::FileNode ocvfnNormImg = ocvfnPreprocess["NormalizeImg"];
    std::vector<float> means = parseFileNodeVector<float>(ocvfnNormImg["mean"]);
    std::vector<float> stds = parseFileNodeVector<float>(ocvfnNormImg["std"]);
    imagePreTransforms.normalize = NormalizeImg{
        {means[0], means[1], means[2]},
        {stds[0], stds[1], stds[2]}
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
    targetPostTransforms.filterBoxes = FilterBoxes{thresh};
    // ConvertBox
    cv::FileNode ocvfnConvertBox = ocvfnPreprocess["ConvertBox"];
    BoxFormat fmt = mapFileNodeString<BoxFormat>(ocvfnConvertBox["src_fmt"]);
    targetPostTransforms.convert = ConvertBox{fmt};
    // RescaleBox
    cv::FileNode ocvfnRescaleBox = ocvfnPreprocess["RescaleBox"];
    std::vector<float> offset = parseFileNodeVector<float>(ocvfnRescaleBox["offset"]);
    std::vector<float> scale = parseFileNodeVector<float>(ocvfnRescaleBox["scale"]);
    targetPostTransforms.rescale = RescaleBox{
        {offset[0], offset[1]},
        {scale[0], scale[1]}
    };
    // ResizeBox
    cv::FileNode ocvfnResizeBox = ocvfnPreprocess["ResizeBox"];
    std::vector<int> size = parseFileNodeVector<int>(ocvfnResizeBox["size"]);
    ResizeMethod resizeMethod = mapFileNodeString<ResizeMethod>(ocvfnResizeBox["method"]);
    targetPostTransforms.resize = ResizeBox{
        {size[0], size[1]},
        resizeMethod
    };
    // NMS
    cv::FileNode ocvfnNMS = ocvfnPreprocess["NMS"];
    float maxOverlap = parseFileNodeValue<float>(ocvfnNMS["max_overlap"]);
    float scaleFactor = parseFileNodeValue<float>(ocvfnNMS["nms_scale_factor"]);
    float outScaleFactor = parseFileNodeValue<float>(ocvfnNMS["output_scale_factor"]);
    targetPostTransforms.nms = NMS{maxOverlap, scaleFactor, outScaleFactor};

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

void CfgParser::mSetImgSize(const cv::Size& size){
    this->aTargetPostTransforms.resize.size = size;
}

const cv::Size CfgParser::mGetImgSize(){
    return this->aTargetPostTransforms.resize.size;
}
