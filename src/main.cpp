#include <chrono>
#include <opencv2/cudaimgproc.hpp>

#include "engine.h"
#include "argparseUtils.h"
#include "inferenceParams.h"


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
    std::array<float, 3> imgSubVals;
    std::array<float, 3> imgDivVals;
    std::copy_n(imagePreTransforms.normalize.mean.begin(), 3, imgSubVals.begin());
    std::copy_n(imagePreTransforms.normalize.std.begin(), 3, imgDivVals.begin());
    //TODO: normalization should be done in a separate function as for postprocessing
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

    // ********************
    // Image pre-processing
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
                float rx = static_cast<float>(targetPostTransforms.invertResize.width)/static_cast<float>(imagePreTransforms.resize.width);
                float ry = static_cast<float>(targetPostTransforms.invertResize.height)/static_cast<float>(imagePreTransforms.resize.height);
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

    // **********************
    // Target post-processing
    // **********************

    const auto& outputDims = engine.getOutputDims();
    unsigned int featureBboxIdx = 0;
    unsigned int featureConfsIdx = 1;
    const auto& boxesShape = outputDims[featureBboxIdx];
    const auto& confsShape = outputDims[featureConfsIdx];
    size_t numAnchors = boxesShape.d[1];
    size_t numClasses = confsShape.d[2];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> nmsIndices;

    // Get all the YOLO proposals
    for (size_t i = 0; i < numAnchors; i++) {
        // Get bbox info
        int batchId = 0;
        const auto currBboxPtr = &featureVectors[batchId][featureBboxIdx][i*4];
        const auto currScoresPtr = &featureVectors[batchId][featureConfsIdx][i*numClasses];
        auto bestScorePtr = std::max_element(currScoresPtr, currScoresPtr+numClasses);
        float bestScore = *bestScorePtr;
        // convert to cv::Rect_ format
        float x, y, w, h;
        if (targetPostTransforms.boxConvert.srcFmt == "cxcywh"){
            x = (*currBboxPtr - *(currBboxPtr+2) / 2.f);
            y = (*(currBboxPtr+1) - *(currBboxPtr+3) / 2.f);
            w = *(currBboxPtr+2);
            h = *(currBboxPtr+3);
        }
        else{
            std::cout << "Box conversion other than \"cxcywh\" not implemented yet!" << imagePath << std::endl;
            return -1;
        }
        // invert normalize bbox from [0.0 - 1.0] to [0.0 - netwh]
        auto subVals = targetPostTransforms.invertNormalize.mean;
        auto divVals = targetPostTransforms.invertNormalize.std;
        x = x * divVals[0] + subVals[0];
        y = y * divVals[1] + subVals[1];
        w = w * divVals[0] + subVals[0];
        h = h * divVals[1] + subVals[1];
        // resize bboxes
        float rx = static_cast<float>(targetPostTransforms.invertResize.width)/static_cast<float>(imagePreTransforms.resize.width);
        float ry = static_cast<float>(targetPostTransforms.invertResize.height)/static_cast<float>(imagePreTransforms.resize.height);
        if (targetPostTransforms.invertResize.method == "maintain_ar"){
                rx = std::max(rx, ry);
                ry = rx;
        }
        x = x * rx;
        y = y * ry;
        w = w * rx;
        h = h * ry;
        // get label and instanciate bbox
        int label = bestScorePtr - currScoresPtr;
        cv::Rect2f bbox;
        bbox.x = x;
        bbox.y = y;
        bbox.width = w;
        bbox.height = h;

        bboxes.push_back(bbox);
        labels.push_back(label);
        scores.push_back(bestScore);
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, targetPostTransforms.nms.threshold, targetPostTransforms.nms.maxOverlap, nmsIndices);
    std::vector<AnnotItem> annotItems;
    for (auto& currIdx : nmsIndices) {
        AnnotItem item{};
        item.probability = scores[currIdx];
        item.label = labels[currIdx];
        item.rect = bboxes[currIdx];
        annotItems.push_back(item);
    }

    // save annotated image
    drawObjectLabels(cpuImg, annotItems, 1.0);
    std::filesystem::path outputImagePath = Util::getDirPath(imagePath);
    outputImagePath = outputImagePath.append("annotated.jpg");
    cv::imwrite(outputImagePath, cpuImg);
    std::cout << "Saved annotated image to: " << outputImagePath << std::endl;

    return 0;
}
