#include <numeric>
#include <utility>
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "detector.h"

#ifdef WITH_BENCHMARK
    #include "utils.h"
#endif


Detector::Detector(const std::filesystem::path& onnxModelPath, const std::filesystem::path& cfgPath){
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
    cv::Size netSize = {inpDims[0].d[1], inpDims[0].d[2]};
    this->mSetNetSize(netSize);
}

const std::vector<std::string> Detector::mGetLabels(){
    return this->aConfig.aLabels;
}

const std::vector<std::vector<float>> Detector::mGetColors(){
    return this->aConfig.aColors;
}

void Detector::mSetNetSize(const cv::Size& size){
    this->aNetSize = size;
    // loop through all transform and set size and scale
    for (auto& ImgTransform: this->aConfig.aImagePreTransforms){
        if(ImgTransform->aSetSize){
            std::dynamic_pointer_cast<Transforms::ResizeImg>(ImgTransform)->mSetSize(size);
        }
    }
    // for (const auto& BBoxTransform: this->aConfig.aBBoxVecPreTransforms){
    //     if(BBoxTransform->aSetScale){
    //         (Transforms::RescaleBBox*) BBoxTransform->mSetScale(
    //             cv::Vec2f(static_cast<float>(size.width),
    //             static_cast<float>(size.height))
    //         );
    //     }
    // }
    this->aConfig.aTargetPostTransforms.rescale.mSetScale(
        cv::Vec2f(static_cast<float>(size.width), static_cast<float>(size.height))
    );
}

void Detector::mSetImgSize(const cv::Size& size){
    this->aImgSize = size;
    // loop through all transform and set size
    // for (const auto& BBoxTransform: this->aConfig.aBBoxVecPreTransforms){
    //     if(BBoxTransform->aSetScale){
    //         (Transforms::ResizeBBox*) BBoxTransform->mSetSize(size);
    //     }
    // }
    this->aConfig.aTargetPostTransforms.resize.mSetSize(size);
}

const std::vector<std::vector<cv::cuda::GpuMat>> Detector::mPreProcess(const cv::cuda::GpuMat& gpuImg){
    // Loop through all inputs
    // Standard detector (which is the case here) should have only one input
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    const auto& inputDims = this->aEngine->getInputDims();
    const int32_t optBatchSize = this->aOptions.optBatchSize;

    for (const auto& inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (int32_t j = 0; j < optBatchSize; ++j) {
            cv::cuda::GpuMat cudaImg = gpuImg;
            for(const auto& currTransform: this->aConfig.aImagePreTransforms){
                cudaImg = currTransform->run(cudaImg);
            }
            input.emplace_back(std::move(cudaImg));
        }
        inputs.emplace_back(std::move(input));
    }

    return inputs;
}

const std::vector<BoundingBox> Detector::mPostProcess(const std::vector<std::vector<std::vector<float>>>& features){
    const auto& outputDims = this->aEngine->getOutputDims();
    unsigned int featureBBoxId = 0;
    unsigned int featureProbsId = 1;
    unsigned int featureConfsId = 2;
    const auto& boxesShape = outputDims[featureBBoxId];
    const auto& probsShape = outputDims[featureProbsId];
    size_t batchSize = this->aOptions.optBatchSize;
    size_t numBBoxes =  boxesShape.d[1];
    size_t numClasses = probsShape.d[2];

    // initialize BBoxes list
    //TODO init takes lot of times, to optimize.
    std::vector<BoundingBox> listBBoxes;
    for (size_t batchId = 0; batchId < batchSize; batchId++){
        for (size_t boxId = 0; boxId < numBBoxes; boxId++){
            // bbox from trt output
            const auto currBboxPtr = &features[batchId][featureBBoxId][boxId*4];
            const auto currProbPtr = &features[batchId][featureProbsId][boxId*numClasses];
            const auto currConfPtr = &features[batchId][featureConfsId][boxId];
            auto bestProbPtr = std::max_element(currProbPtr, currProbPtr+numClasses);
            float bestProb = *bestProbPtr;
            float conf = *currConfPtr;
            const cv::Vec4f inp = {*currBboxPtr, *(currBboxPtr+1), *(currBboxPtr+2), *(currBboxPtr+3)};
            float score = conf * bestProb;
            int label = bestProbPtr - currProbPtr;
            // fill-in the detections
            BoundingBox inpBBox(inp, this->aNetSize, this->aConfig.aBBoxSrcFormat, label, score);
            listBBoxes.emplace_back(inpBBox);
        }
    }
    auto& targetPostTransforms = this->aConfig.aTargetPostTransforms;
    // filter-out bad boxes
    const std::vector<BoundingBox> filteredBBoxes = targetPostTransforms.filter.run(listBBoxes);
    // then loop through all valid bboxes and transform
    std::vector<BoundingBox> outBBoxes;
    for (const auto& bbox: filteredBBoxes) {
        // box post processing
        const BoundingBox converted = targetPostTransforms.convert.run(bbox);
        const BoundingBox rescaled = targetPostTransforms.rescale.run(converted);
        const BoundingBox resized = targetPostTransforms.resize.run(rescaled);
        // append post-processed bbox
        outBBoxes.push_back(resized);
    }
    // run NMS
    const std::vector<BoundingBox> NMSedBBoxes = targetPostTransforms.nms.run(outBBoxes);

    return NMSedBBoxes;
}

const std::vector<BoundingBox> Detector::mPredict(const cv::cuda::GpuMat& gpuImg){

    this->mSetImgSize(gpuImg.size());  //used by post processing
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
