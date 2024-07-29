#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <fstream>

#include "detector.h"
#include "argparseUtils.h"

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

    const std::filesystem::path modelPath = program.get<std::string>("--model");
    const std::filesystem::path cfgPath = program.get<std::string>("--cfg");
    const std::filesystem::path imagePath = program.get<std::string>("--image");
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
    // instanciate detector from config and ONNX model
    Detector detector(modelPath, cfgPath);

    // Read image and make inference
    auto cpuImg = cv::imread(imagePath);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + std::string(imagePath));
    }
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);
    preciseStopwatch stopwatch;
    const std::vector<AnnotItem> annotItems = detector.mPredict(gpuImg);
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    std::cout << "total:" << totalElapsedTimeMs << std::endl;

    // save annotated image
    const CfgParser cfgparser = detector.mGetConfig();
    drawObjectLabels(cpuImg, annotItems, 1.0, cfgparser);
    std::filesystem::path outputImagePath = Util::getDirPath(imagePath);
    outputImagePath = outputImagePath.append("annotated.jpg");
    cv::imwrite(outputImagePath, cpuImg);
    std::cout << "Saved annotated image to: " << outputImagePath << std::endl;

    return 0;
}
