#include <chrono>
#include <opencv2/cudaimgproc.hpp>

#include "engine.h"
#include "argparseUtils.h"
#include "inferenceParams.h"


int main(int argc, char *argv[]) {
    // Parse the command line arguments
    argparse::ArgumentParser program(argv[0], argparseUtils::appVersion);
    argparseUtils::setArgParser(program);
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::filesystem::path modelPath = program.get<std::string>("--model");
    std::filesystem::path imagePath = program.get<std::string>("--image");
    // Ensure the model and image files exists
    if (!std::filesystem::exists(modelPath)) {
        std::cout << "Error: Unable to find file at path: " << modelPath << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(imagePath)) {
        std::cout << "Error: Unable to find file at path: " << imagePath << std::endl;
        return -1;
    }

    // **************************
    // Model loading and building
    // **************************

    // Specify our GPU inference configuration options
    Options options;
    options.precision = Precision::FP16; // Specify what precision to use for inference.
    options.calibrationDataDirectoryPath = ""; // If using INT8 precision, must use calibration data.
    options.optBatchSize = 1; // Specify the batch size to optimize for.
    options.maxBatchSize = 1; // Specify the maximum batch size we plan on running.
    Engine engine(options);

    // Define pre-processing options
    //TODO: initialization from file `models/params.yaml`
    inferenceParams::ImagePreTransforms imagePreTransforms;
    inferenceParams::TargetPostTransforms targetPostTransforms;

    // Build the onnx model into a TensorRT engine file.
    bool succ = engine.build(modelPath,
        imagePreTransforms.normalize.mean,
        imagePreTransforms.normalize.std,
        imagePreTransforms.toDtype.scale);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // ********************
    // Input pre-processing
    // ********************

    // Read the input image
    auto cpuImg = cv::imread(imagePath);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + std::string(imagePath));
    }

    //TODO: Define an invert transform for resizing
    // Post processing invertResize takes input image size
    targetPostTransforms.invertResize.method = imagePreTransforms.resize.method;
    targetPostTransforms.invertResize.height = cpuImg.rows;
    targetPostTransforms.invertResize.width = cpuImg.cols;

    // Upload the image GPU memory and convert from BGR to RGB
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);
    cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

    // In the following section we populate the input vectors to later pass for inference
    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    auto resized = gpuImg;
    // Loop through all inputs, standard detector (which is the case here) should have only one
    for (const auto & inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (auto j = 0; j < options.optBatchSize; ++j) {
            if (imagePreTransforms.resize.method == "maintain_ar"){
                resized = Engine::resizeKeepAspectRatioPadRightBottom(
                    gpuImg, imagePreTransforms.resize.height, imagePreTransforms.resize.width);
            } else if (imagePreTransforms.resize.method == "scale"){
                cv::cuda::resize(gpuImg, resized, cv::Size(
                    imagePreTransforms.resize.width, imagePreTransforms.resize.height));
            }
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    #ifndef NDEBUG
        cv::Mat preprocImage;
        inputs[0][0].download(preprocImage);
        cv::cvtColor(preprocImage, preprocImage, cv::COLOR_RGB2BGR);
        cv::imwrite("test.png", preprocImage);

        // check post process
        for (const auto & inputDim : inputDims) {
            for (auto j = 0; j < options.optBatchSize; ++j) {
                cv::Mat outImg;
                cv::cuda::GpuMat gpuImg = inputs[0][j];
                float rx = static_cast<float>(targetPostTransforms.invertResize.width)/static_cast<float>(gpuImg.cols);
                float ry = static_cast<float>(targetPostTransforms.invertResize.height)/static_cast<float>(gpuImg.rows);
                if (targetPostTransforms.invertResize.method == "maintain_ar"){
                    rx = std::max(rx, ry);
                    ry = rx;
                }
                cv::cuda::resize(gpuImg, gpuImg, cv::Size(), rx, ry);
                if (targetPostTransforms.invertResize.method == "maintain_ar"){
                    cv::Rect myROI(0, 0, targetPostTransforms.invertResize.width, targetPostTransforms.invertResize.height);
                    gpuImg = gpuImg(myROI);
                }
                gpuImg.download(outImg);
                cv::cvtColor(outImg, outImg, cv::COLOR_RGB2BGR);
                cv::imwrite("test_invert_preproc.png", outImg);
            }
        }
    #endif

    std::vector<std::vector<std::vector<float>>> featureVectors;
    engine.runInference(inputs, featureVectors);

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e:  featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }
    }

//     // TODO: If your model requires post processing (ex. convert feature vector into bounding boxes) then you would do so here.

    return 0;
}
