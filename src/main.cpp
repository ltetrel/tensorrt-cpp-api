#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <fstream>

#include "engine.h"
#include "argparseUtils.h"
#include "inferenceParams.h"
#include "inferenceParams2.h"
#include "transforms.h"


namespace {

void drawObjectLabels(cv::Mat &image, const std::vector<AnnotItem> &objects, unsigned int scale) {
    // Bounding boxes and annotations
    for (auto &object : objects) {
        // Choose the color
        int colorIndex = object.label % inferenceParams::colors.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(inferenceParams::colors[colorIndex][0], inferenceParams::colors[colorIndex][1], inferenceParams::colors[colorIndex][2]);
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
        sprintf(text, "%s %.1f%%", inferenceParams::classLabels[object.label].c_str(), object.probability * 100);

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
    //TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/index.html
    // https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/index.html

    // Specify our GPU inference configuration options
    Options options;
    options.precision = Precision::FP16; // Specify what precision to use for inference.
    options.calibrationDataDirectoryPath = ""; // If using INT8 precision, must use calibration data.
    options.optBatchSize = 1; // Specify the batch size to optimize for.
    options.maxBatchSize = 1; // Specify the maximum batch size we plan on running.
    Engine engine(options);

    // Define pre-processing options
    //TODO: initialization from file `models/inference_params.yaml`
    inferenceParams::ImagePreTransforms imagePreTransforms;
    inferenceParams::TargetPostTransforms targetPostTransforms;
    inferenceParams2::ImagePreTransforms imagePreTransformsv2;
    inferenceParams2::TargetPostTransforms targetPostTransformsv2;

    // Build the onnx model into a TensorRT engine file.
    std::array<float, 3> imgSubVals;
    std::array<float, 3> imgDivVals;
    std::copy_n(imagePreTransforms.normalize.mean.begin(), 3, imgSubVals.begin());
    std::copy_n(imagePreTransforms.normalize.std.begin(), 3, imgDivVals.begin());
    bool succ = engine.build(modelPath,
        imgSubVals,
        imgDivVals,
        imagePreTransforms.toDtype.scale);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    // Load the TensorRT engine file from disk
    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    preciseStopwatch stopwatch;

    // ********************
    // Image pre-processing
    // ********************

    // Read the input image
    auto cpuImg = cv::imread(imagePath);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + std::string(imagePath));
    }

    // Post processing box resizing takes input image size
    targetPostTransformsv2.resize.tgtSize = cpuImg.size();
    targetPostTransformsv2.resize.method = imagePreTransformsv2.resize.method;

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
                colored, imagePreTransformsv2.resize.tgtSize, imagePreTransformsv2.resize.method);
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    #ifndef NDEBUG
    cv::Mat preprocImage;
    inputs[0][0].download(preprocImage);
    cv::imwrite("preprocImage.png", preprocImage);
    #endif


    std::vector<std::vector<std::vector<float>>> featureVectors;
    //TODO: normalization should be done in a separate function as for postprocessing
    engine.runInference(inputs, featureVectors);

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

    inferenceParams::Model model;
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

    if (model.type == "darknet"){
        validBoxIds = Transforms::getValidBoxIds(
            featureVectors[batchId][featureConfsIdx], targetPostTransformsv2.filterBoxes.thresh);
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
        const cv::Vec4f converted = Transforms::convertBox(inp, targetPostTransformsv2.convert.srcFmt);
        const cv::Vec4f rescaled = Transforms::rescaleBox(
            converted, targetPostTransformsv2.rescale.offset, targetPostTransformsv2.rescale.scale);
        const cv::Vec4f resized = Transforms::resizeBox(
            rescaled,
            targetPostTransformsv2.resize.inpSize,
            targetPostTransformsv2.resize.tgtSize,
            targetPostTransformsv2.resize.method);
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
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, 0.0, targetPostTransformsv2.nms.maxOverlap, nmsIndices);
    std::vector<AnnotItem> annotItems;
    for (auto& currIdx : nmsIndices) {
        AnnotItem item{};
        item.probability = scores[currIdx];
        item.label = labels[currIdx];
        item.rect = bboxes[currIdx];
        annotItems.push_back(item);
    }

    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    std::cout << totalElapsedTimeMs << std::endl;

    // save annotated image
    drawObjectLabels(cpuImg, annotItems, 1.0);
    std::filesystem::path outputImagePath = Util::getDirPath(imagePath);
    outputImagePath = outputImagePath.append("annotated.jpg");
    cv::imwrite(outputImagePath, cpuImg);
    std::cout << "Saved annotated image to: " << outputImagePath << std::endl;

    return 0;
}
