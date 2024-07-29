#include "detector.h"


Detector::Detector(const std::filesystem::path onnxModelPath, const std::filesystem::path cfgPath){
    // **************************
    // Model loading and building
    // **************************

    // Specify TensorRT engine options
    Options options;
    options.precision = Precision::FP16; // Specify what precision to use for inference.
    options.calibrationDataDirectoryPath = ""; // If using INT8 precision, must use calibration data.
    options.optBatchSize = 1; // Specify the batch size to optimize for.
    options.maxBatchSize = 1; // Specify the maximum batch size we plan on running.
    this->aOptions = options;

    // Parse inference config
    this->aConfig = CfgParser(cfgPath);

    // Build and load the onnx model into a TensorRT engine file
    this->aEngine = std::make_unique<Engine>(options);
    bool succ = this->aEngine->build(onnxModelPath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }
    succ = this->aEngine->loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }
}

const CfgParser Detector::mGetConfig(){
    return this->aConfig;
}

const std::vector<std::vector<cv::cuda::GpuMat>> Detector::mPreProcess(const cv::cuda::GpuMat& gpuImg){
    // Loop through all inputs
    // Standard detector (which is the case here) should have only one input
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    const ImagePreTransforms imagePreTransforms = this->aConfig.aImagePreTransforms;
    const auto& inputDims = this->aEngine->getInputDims();
    const int32_t optBatchSize = this->aOptions.optBatchSize;

    for (const auto& inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (int32_t j = 0; j < optBatchSize; ++j) {
            const cv::cuda::GpuMat colored = Transforms::convertColorImg(gpuImg, imagePreTransforms.convertColor.model);
            const cv::cuda::GpuMat resized = Transforms::resizeImg(
                colored, imagePreTransforms.resize.tgtSize, imagePreTransforms.resize.method);
            const cv::cuda::GpuMat casted = Transforms::castImg(
                resized, imagePreTransforms.cast.dtype, imagePreTransforms.cast.scale);
            const cv::cuda::GpuMat normalized = Transforms::normalizeImg(
                casted, imagePreTransforms.normalize.mean, imagePreTransforms.normalize.std);
            input.emplace_back(std::move(normalized));
        }
        inputs.emplace_back(std::move(input));
    }

    return inputs;
}

const std::vector<AnnotItem> Detector::mPostProcess(const std::vector<std::vector<std::vector<float>>>& features){
    const TargetPostTransforms targetPostTransforms = this->aConfig.aTargetPostTransforms;
    const auto& outputDims = this->aEngine->getOutputDims();
    unsigned int featureBboxIdx = 0;
    unsigned int featureProbsIdx = 1;
    unsigned int featureConfsIdx = 2;
    const auto& boxesShape = outputDims[featureBboxIdx];
    const auto& probsShape = outputDims[featureProbsIdx];

    size_t numAnchors = boxesShape.d[1];
    size_t numClasses = probsShape.d[2];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> nmsIndices;
    int batchId = 0;  //TODO: manage batch size

    // initialize valid bboxes from range
    std::vector<unsigned int> validBoxIds(features[batchId][featureConfsIdx].size());
    std::iota(validBoxIds.begin(), validBoxIds.end(), 0);

    if (this->aConfig.aModel.type == ModelType::darknet){
        validBoxIds = Transforms::getValidBoxIds(
            features[batchId][featureConfsIdx], targetPostTransforms.filterBoxes.thresh);
    }

    // loop through all valid boxes
    for (const unsigned int validBoxId: validBoxIds) {
        // Get current bbox info
        const auto currBboxPtr = &features[batchId][featureBboxIdx][validBoxId*4];
        const auto currProbPtr = &features[batchId][featureProbsIdx][validBoxId*numClasses];
        const auto currConfPtr = &features[batchId][featureConfsIdx][validBoxId];
        auto bestProbPtr = std::max_element(currProbPtr, currProbPtr+numClasses);
        float bestProb = *bestProbPtr;
        float conf = *currConfPtr;
        const cv::Vec4f inp = {*currBboxPtr, *(currBboxPtr+1), *(currBboxPtr+2), *(currBboxPtr+3)};
        float score = conf * bestProb;
        int label = bestProbPtr - currProbPtr;
        // box post processing
        const cv::Vec4f converted = Transforms::convertBox(inp, targetPostTransforms.convert.srcFmt);
        const cv::Vec4f rescaled = Transforms::rescaleBox(
            converted, targetPostTransforms.rescale.offset, targetPostTransforms.rescale.scale);
        const cv::Vec4f resized = Transforms::resizeBox(
            rescaled,
            targetPostTransforms.resize.inpSize,
            targetPostTransforms.resize.tgtSize,
            targetPostTransforms.resize.method);
        // output post-processed bbox
        cv::Rect2f bbox;
        bbox.x = resized[0];
        bbox.y = resized[1];
        bbox.width = resized[2];
        bbox.height = resized[3];

        bboxes.push_back(bbox);
        labels.push_back(label);
        scores.push_back(score);
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, 0.0, targetPostTransforms.nms.maxOverlap, nmsIndices);
    std::vector<AnnotItem> annotItems;
    for (auto& currIdx : nmsIndices) {
        AnnotItem item{};
        item.probability = scores[currIdx];
        item.label = labels[currIdx];
        item.rect = bboxes[currIdx];
        annotItems.push_back(item);
    }

    return annotItems;
}

const std::vector<AnnotItem> Detector::mPredict(const cv::cuda::GpuMat& gpuImg){
    // Image pre-processing
    const std::vector<std::vector<cv::cuda::GpuMat>> inputs = mPreProcess(gpuImg);

    // Inference
    std::vector<std::vector<std::vector<float>>> features;
    this->aEngine->runInference(inputs, features);

    // Target post-processing
    //TODO: use boundingBox class to manage img size thourgh canvasSize attribute.
    this->aConfig.mSetImgSize(gpuImg.size());
    const std::vector<AnnotItem> annotItems = mPostProcess(features);

    return annotItems;
}
