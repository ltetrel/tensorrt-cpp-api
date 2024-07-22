#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <fstream>

#include "engine.h"
#include "argparseUtils.h"
#include "transforms.h"
#include "configParser.h"


namespace {

void drawObjectLabels(cv::Mat &image, const std::vector<AnnotItem> &objects, unsigned int scale, CfgParser cfgParser) {
    // Bounding boxes and annotations
    for (auto &object : objects) {
        // Choose the color
        int colorIndex = object.label % cfgParser.aColors.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(cfgParser.aColors[colorIndex][0], cfgParser.aColors[colorIndex][1], cfgParser.aColors[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5) {
            txtColor = cv::Scalar(0, 0, 0);
        } else {
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto &rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", cfgParser.aLabels[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
    }
}

}

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
    std::filesystem::path cfgPath = program.get<std::string>("--cfg");
    std::filesystem::path imagePath = program.get<std::string>("--image");
    // Ensure the model and image files exists
    if (!std::filesystem::exists(modelPath)) {
        std::cout << "Error: Unable to find model at path: " << modelPath << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(cfgPath)) {
        std::cout << "Error: Unable to find pipe at path: " << cfgPath << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(imagePath)) {
        std::cout << "Error: Unable to find image at path: " << imagePath << std::endl;
        return -1;
    }

    // **************************
    // Model loading and building
    // **************************
    //TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/index.html
    // https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/index.html

    // Specify our GPU inference configuration options
    Options options;
    options.precision = Precision::FP16; // Specify what precision to use for inference.
    options.calibrationDataDirectoryPath = ""; // If using INT8 precision, must use calibration data.
    options.optBatchSize = 1; // Specify the batch size to optimize for.
    options.maxBatchSize = 1; // Specify the maximum batch size we plan on running.
    Engine engine(options);

    // Parse inference config
    cfgPath = "models/inference_params.yaml";
    CfgParser cfgParser(cfgPath);

    // Build the onnx model into a TensorRT engine file.
    bool succ = engine.build(modelPath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // ********************
    // Image pre-processing
    // ********************

    // Read the input image
    auto cpuImg = cv::imread(imagePath);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + std::string(imagePath));
    }

    // Post processing box resizing takes input image size
    cfgParser.mSetImgSize(cpuImg.size());

    preciseStopwatch stopwatch;

    // Upload the image GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);

    // In the following section we populate the input vectors to later pass for inference
    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Loop through all inputs, standard detector (which is the case here) should have only one
    for (const auto & inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (auto j = 0; j < options.optBatchSize; ++j) {
            const cv::cuda::GpuMat colored = Transforms::convertColorImg(gpuImg, ColorModel::BGR);
            const cv::cuda::GpuMat resized = Transforms::resizeImg(
                colored, cfgParser.aImagePreTransforms.resize.tgtSize, cfgParser.aImagePreTransforms.resize.method);
            const cv::cuda::GpuMat casted = Transforms::castImg(
                resized, cfgParser.aImagePreTransforms.cast.dtype, cfgParser.aImagePreTransforms.cast.scale);
            const cv::cuda::GpuMat normalized = Transforms::normalizeImg(
                casted, cfgParser.aImagePreTransforms.normalize.mean, cfgParser.aImagePreTransforms.normalize.std);
            input.emplace_back(std::move(normalized));
        }
        inputs.emplace_back(std::move(input));
    }

    #ifndef NDEBUG
    cv::Mat preprocImage;
    inputs[0][0].download(preprocImage);
    cv::imwrite("preprocImage.png", preprocImage);
    #endif

    auto elps1 = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    std::cout << "preproc: " << elps1 << std::endl;

    std::vector<std::vector<std::vector<float>>> featureVectors;
    engine.runInference(inputs, featureVectors);

    auto elps2 = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    std::cout << "infer: " << elps2 << std::endl;

    #ifndef NDEBUG
    std::ofstream boxes_file("./inputs/boxes.txt");
    std::ofstream labels_file("./inputs/labels.txt");
    std::ostream_iterator<float> output_iterator_boxes(boxes_file, "\n");
    std::ostream_iterator<float> output_iterator_labels(labels_file, "\n");
    std::copy(featureVectors[0][0].begin(), featureVectors[0][0].end(), output_iterator_boxes);
    std::copy(featureVectors[0][1].begin(), featureVectors[0][1].end(), output_iterator_labels);
    #endif

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

    // **********************
    // Target post-processing
    // **********************

    const auto& outputDims = engine.getOutputDims();
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
    std::vector<unsigned int> validBoxIds(featureVectors[batchId][featureConfsIdx].size());
    std::iota(validBoxIds.begin(), validBoxIds.end(), 0);

    if (cfgParser.aModel.type == ModelType::darknet){
        validBoxIds = Transforms::getValidBoxIds(
            featureVectors[batchId][featureConfsIdx], cfgParser.aTargetPostTransforms.filterBoxes.thresh);
    }

    // loop through all valid boxes
    for (const unsigned int validBoxId: validBoxIds) {
        // Get current bbox info
        const auto currBboxPtr = &featureVectors[batchId][featureBboxIdx][validBoxId*4];
        const auto currProbPtr = &featureVectors[batchId][featureProbsIdx][validBoxId*numClasses];
        const auto currConfPtr = &featureVectors[batchId][featureConfsIdx][validBoxId];
        auto bestProbPtr = std::max_element(currProbPtr, currProbPtr+numClasses);
        float bestProb = *bestProbPtr;
        float conf = *currConfPtr;
        const cv::Vec4f inp = {*currBboxPtr, *(currBboxPtr+1), *(currBboxPtr+2), *(currBboxPtr+3)};
        float score = conf * bestProb;
        int label = bestProbPtr - currProbPtr;
        // box post processing
        const cv::Vec4f converted = Transforms::convertBox(inp, cfgParser.aTargetPostTransforms.convert.srcFmt);
        const cv::Vec4f rescaled = Transforms::rescaleBox(
            converted, cfgParser.aTargetPostTransforms.rescale.offset, cfgParser.aTargetPostTransforms.rescale.scale);
        const cv::Vec4f resized = Transforms::resizeBox(
            rescaled,
            cfgParser.aTargetPostTransforms.resize.inpSize,
            cfgParser.aTargetPostTransforms.resize.tgtSize,
            cfgParser.aTargetPostTransforms.resize.method);
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
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, 0.0, cfgParser.aTargetPostTransforms.nms.maxOverlap, nmsIndices);
    std::vector<AnnotItem> annotItems;
    for (auto& currIdx : nmsIndices) {
        AnnotItem item{};
        item.probability = scores[currIdx];
        item.label = labels[currIdx];
        item.rect = bboxes[currIdx];
        annotItems.push_back(item);
    }

    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    std::cout << "total:" << totalElapsedTimeMs << std::endl;

    // save annotated image
    drawObjectLabels(cpuImg, annotItems, 1.0, cfgParser);
    std::filesystem::path outputImagePath = Util::getDirPath(imagePath);
    outputImagePath = outputImagePath.append("annotated.jpg");
    cv::imwrite(outputImagePath, cpuImg);
    std::cout << "Saved annotated image to: " << outputImagePath << std::endl;

    return 0;
}
