#include <numeric>
#include <utility>
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "detector.h"

#ifdef WITH_BENCHMARK
    #include "utils.h"
#endif


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

    // Get net shape and update transforms 
    const auto& inpDims = this->aEngine->getInputDims();
    this->aNetShape = {inpDims[0].d[1], inpDims[0].d[2]};
    this->aConfig.aImagePreTransforms.resize.size = {inpDims[0].d[1], inpDims[0].d[2]};
    this->aConfig.aTargetPostTransforms.rescale.scale = {
        static_cast<float>(inpDims[0].d[1]),
        static_cast<float>(inpDims[0].d[2])
    };
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
                colored, imagePreTransforms.resize.size, imagePreTransforms.resize.method);
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

const std::vector<BoundingBox> Detector::mPostProcess(const std::vector<std::vector<std::vector<float>>>& features){
    const TargetPostTransforms targetPostTransforms = this->aConfig.aTargetPostTransforms;
    const auto& outputDims = this->aEngine->getOutputDims();
    unsigned int featureBboxIdx = 0;
    unsigned int featureProbsIdx = 1;
    unsigned int featureConfsIdx = 2;
    // const auto& boxesShape = outputDims[featureBboxIdx];
    const auto& probsShape = outputDims[featureProbsIdx];

    // size_t numAnchors = boxesShape.d[1];
    size_t numClasses = probsShape.d[2];

    std::vector<BoundingBox> listBBoxes;
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> nmsIndices;
    int batchId = 0;  //TODO: manage batch size

    // Get input bbox format
    BoxFormat inpBoxFormat;
    if (this->aConfig.aModel.type == ModelType::darknet){
        inpBoxFormat = BoxFormat::cxcywh;
    }
    else if (this->aConfig.aModel.type == ModelType::netharn){
        inpBoxFormat = BoxFormat::xywh;
    }
    else{
        inpBoxFormat = targetPostTransforms.convert.srcFmt;  // by default use src format from config
    }

    // filter-out bad boxes
    std::vector<unsigned int> validBoxIds(features[batchId][featureConfsIdx].size());
    std::iota(validBoxIds.begin(), validBoxIds.end(), 0);
    validBoxIds = Transforms::getValidBoxIds(
        features[batchId][featureConfsIdx], targetPostTransforms.filterBoxes.thresh);
    // then loop through all valid boxes
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
        const BoundingBox inpBBox(inp, this->aNetShape, inpBoxFormat, label, score);
        // box post processing
        const BoundingBox converted = Transforms::convertBBox(inpBBox, BoxFormat::xywh);
        const BoundingBox rescaled = Transforms::rescaleBBox(
            converted, targetPostTransforms.rescale.offset, targetPostTransforms.rescale.scale);
        const BoundingBox resized = Transforms::resizeBBox(
            rescaled,
            targetPostTransforms.resize.size,
            targetPostTransforms.resize.method);
        // append post-processed bbox
        listBBoxes.push_back(resized);

    }
    // run NMS
    const std::vector<BoundingBox> listBBoxesNMSed = Transforms::nmsBBox(
        listBBoxes,
        targetPostTransforms.nms.maxOverlap,
        targetPostTransforms.nms.nmsScaleFactor,
        targetPostTransforms.nms.outputScaleFactor);

    return listBBoxesNMSed;
}

const std::vector<BoundingBox> Detector::mPredict(const cv::cuda::GpuMat& gpuImg){

    // Image pre-processing
    #ifdef WITH_BENCHMARK
        Utils::preciseStopwatch preStopwatch;
    #endif
    const std::vector<std::vector<cv::cuda::GpuMat>> inputs = mPreProcess(gpuImg);
    #ifdef WITH_BENCHMARK
        auto preprocElpsInUs = preStopwatch.elapsedTime<float, std::chrono::microseconds>();
        this->aPreTimeInMs.emplace_back(preprocElpsInUs/1000);
    #endif

    // Inference
    #ifdef WITH_BENCHMARK
        Utils::preciseStopwatch inferStopwatch;
    #endif
    std::vector<std::vector<std::vector<float>>> features;
    this->aEngine->runInference(inputs, features);
    #ifdef WITH_BENCHMARK
        auto inferenceElpsInUs = inferStopwatch.elapsedTime<float, std::chrono::microseconds>();
        this->aInferTimeInMs.emplace_back(inferenceElpsInUs/1000);
    #endif

    // Target post-processing
    #ifdef WITH_BENCHMARK
        Utils::preciseStopwatch postStopwatch;
    #endif
    this->aConfig.mSetImgSize(gpuImg.size());
    const std::vector<BoundingBox> detections = mPostProcess(features);
    #ifdef WITH_BENCHMARK
        auto postprocElpsInUs = postStopwatch.elapsedTime<float, std::chrono::microseconds>();
        this->aPostTimeInMs.emplace_back(postprocElpsInUs/1000);
    #endif

    return detections;
}

#ifdef WITH_BENCHMARK
    namespace {
        template<typename T>
        inline cv::Mat matFloatVector(std::vector<T>* vector, int offset){
            return cv::Mat(
                cv::Size(1, vector->size() - offset),
                CV_32FC1,
                vector->data() + offset
            );
        };
    }
    void Detector::mPrintBenchmarkSummary(int warmupSize){
        // compute statistics and ignore first 5 values (warm-up)
        cv::Mat merged[3] = {
            matFloatVector(&(this->aPreTimeInMs), warmupSize),
            matFloatVector(&(this->aInferTimeInMs), warmupSize),
            matFloatVector(&(this->aPostTimeInMs), warmupSize)
        };
        cv::Mat matTimes;
        cv::merge(merged, 3, matTimes);
        cv::Scalar matMeans, matStddevs;
        cv::meanStdDev(matTimes, matMeans, matStddevs);

        std::cout << "======================" << std::endl;
        std::cout << "Benchmarking complete!" << std::endl;
        printf("Pre-processing (ms): %f +/- %f\n", matMeans[0], matStddevs[0]);
        printf("Inference (ms): %f +/- %f\n", matMeans[1], matStddevs[1]);
        printf("Post-processing (ms): %f +/- %f\n", matMeans[2], matStddevs[2]);
        int avgFps = 1000 / (matMeans[0] + matMeans[1] + matMeans[2]);
        printf("Avg FPS: %i\n", avgFps);
        std::cout << "======================" << std::endl;
    }
#endif
