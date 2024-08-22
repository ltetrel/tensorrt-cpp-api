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

const std::vector<cv::Scalar> Detector::mGetColors(){
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
    for (auto& BBoxTransform: this->aConfig.aTargetPostTransforms){
        if(BBoxTransform->aSetScale){
            std::dynamic_pointer_cast<Transforms::RescaleBBox>(BBoxTransform)->mSetScale(
                cv::Vec2f(static_cast<float>(size.width), static_cast<float>(size.height))
            );
        }
    }
}

void Detector::mSetImgSize(const cv::Size& size){
    this->aImgSize = size;
    // loop through all transform and set size
    for (auto& BBoxTransform: this->aConfig.aTargetPostTransforms){
        if(BBoxTransform->aSetSize){
            std::dynamic_pointer_cast<Transforms::ResizeBBox>(BBoxTransform)->mSetSize(size);
        }
    }
}

const std::vector<std::vector<cv::cuda::GpuMat>> Detector::mPreProcess(const cv::cuda::GpuMat& gpuImg){
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    const auto& inputDims = this->aEngine->getInputDims();
    const size_t batchSize = this->aOptions.optBatchSize;

    // loop through all inputs
    // standard detector (which is the case here) should have only one input
    for (const auto& inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (int32_t j = 0; j < batchSize; ++j) {
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

const std::vector<BoundingBox> Detector::mTranformModelOutput(const std::vector<std::vector<std::vector<float>>>& features){
    std::vector<BoundingBox> listBBoxes;
    const auto& outputDims = this->aEngine->getOutputDims();
    const size_t batchSize = this->aOptions.optBatchSize;
    ModelBackend modelBackend = this->aConfig.aModel.backend;
    
    if (modelBackend == ModelBackend::darknet){
        const BoxFormat boxFormat = BoxFormat::cxcywh;
        const unsigned int featureBBoxId = 0;
        const unsigned int featureProbsId = 1;
        const unsigned int featureConfsId = 2;
        const auto& boxesShape = outputDims[featureBBoxId];
        const auto& probsShape = outputDims[featureProbsId];
        const size_t numBBoxes =  boxesShape.d[1];
        const size_t numClasses = probsShape.d[2];

        // initialize BBoxes list
        //TODO init takes lot of times, to optimize.
        for (size_t batchId = 0; batchId < batchSize; batchId++){
            for (size_t boxId = 0; boxId < numBBoxes; boxId++){
                // bbox from trt output
                const auto currBboxPtr = &features[batchId][featureBBoxId][boxId*4];
                const auto currProbPtr = &features[batchId][featureProbsId][boxId*numClasses];
                const auto currConfPtr = &features[batchId][featureConfsId][boxId];
                auto bestProbPtr = std::max_element(currProbPtr, currProbPtr+numClasses);
                float bestProb = *bestProbPtr;
                float conf = *currConfPtr;
                cv::Vec4f bounds = {*currBboxPtr, *(currBboxPtr+1), *(currBboxPtr+2), *(currBboxPtr+3)};
                float score = conf * bestProb;
                int label = bestProbPtr - currProbPtr;
                // fill-in the detections
                BoundingBox inpBBox(bounds, this->aNetSize, boxFormat, label, score);
                listBBoxes.emplace_back(inpBBox);
            }
        }
    }
    else{
        throw std::runtime_error("Transforming output for this model backend is not supported yet!");
    }

    return listBBoxes;
}

const std::vector<BoundingBox> Detector::mPostProcess(const std::vector<std::vector<std::vector<float>>>& features){
    // get BBox list from model output
    std::vector<BoundingBox> listBBoxes = this->mTranformModelOutput(features);
    // filter-out bad boxes
    std::vector<BoundingBox> filteredBBoxes = this->aConfig.aTargetsFilterTransform->run(listBBoxes);
    // then loop through all valid bboxes and transform
    std::vector<BoundingBox> outBBoxes;
    for (BoundingBox& bbox: filteredBBoxes) {
        for(const auto& currTransform: this->aConfig.aTargetPostTransforms){
            bbox = currTransform->run(bbox);
        }
        outBBoxes.push_back(bbox);
    }
    // run NMSBBoxes
    const std::vector<BoundingBox> NMSedBBoxes = this->aConfig.aTargetsNMSTransform->run(outBBoxes);

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
