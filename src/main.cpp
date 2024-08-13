#include <argparse/argparse.hpp>

#include "detector.h"
#include "utils.h"

namespace {

void drawObjectLabels(cv::Mat &image, const std::vector<BoundingBox> &objects, unsigned int scale, CfgParser cfgParser) {
    // Bounding boxes and annotations
    for (auto &object : objects) {
        // Choose the color
        int colorIndex = object.aLabel % cfgParser.aColors.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(cfgParser.aColors[colorIndex][0], cfgParser.aColors[colorIndex][1], cfgParser.aColors[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5) {
            txtColor = cv::Scalar(0, 0, 0);
        } else {
            txtColor = cv::Scalar(255, 255, 255);
        }

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", cfgParser.aLabels[object.aLabel].c_str(), object.aConf * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);
        cv::Scalar txt_bk_color = color * 0.7 * 255;

        const cv::Rect rect = {
            static_cast<int>(object.aBounds[0]),
            static_cast<int>(object.aBounds[1]),
            static_cast<int>(object.aBounds[2]),
            static_cast<int>(object.aBounds[3])
        };
        int x = rect.x;
        int y = rect.y + 1;
        cv::rectangle(image, rect, color * 255, scale + 1);
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
    }
}

bool setArgParser(argparse::ArgumentParser& program){
    program.add_argument("--model")
        .required()
        .help("Path to the model (.onnx or .trt)");
    program.add_argument("--cfg")
        .required()
        .help("Path to the inference config file (.yaml)");
    program.add_argument("--image")
        .required()
        .help("Path to an image (check supported formats here: "
            "https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imread");
    return true;
}

}

int main(int argc, char *argv[]) {
    // parse the command line arguments
    argparse::ArgumentParser program(argv[0], Utils::apiVersion);
    setArgParser(program);
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

    // read image and put on GPU
    auto cpuImg = cv::imread(imagePath);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + std::string(imagePath));
    }
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(cpuImg);

    // instanciate detector then do inference
    Detector detector(modelPath, cfgPath);
    #ifdef WITH_BENCHMARK
        // initial warmup
        for (size_t i = 0; i < 100; ++i) {
            cv::RNG rng(i);
            cv::Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
            cv::Mat sigma = cv::Mat::ones(1,1,CV_64FC1);
            cv::Mat randnMat(gpuImg.size(), gpuImg.type());
            rng.fill(randnMat, cv::RNG::NORMAL, mean, sigma);
            cv::cuda::GpuMat gpuRandn;
            gpuRandn.upload(randnMat);
            std::vector<BoundingBox> detectionsB = detector.mPredict(gpuRandn);
            detectionsB.clear();
        }
        detector.mPrintBenchmarkSummary();
    #endif
    Utils::preciseStopwatch stopwatch;
    const std::vector<BoundingBox> detections = detector.mPredict(gpuImg);
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    std::cout << "Total detection time (ms):" << totalElapsedTimeMs << std::endl;


    // save annotated image
    const CfgParser cfgparser = detector.mGetConfig();
    drawObjectLabels(cpuImg, detections, 1.0, cfgparser);
    std::filesystem::path outputImagePath = Utils::getDirPath(imagePath);
    outputImagePath = outputImagePath.append("annotated.jpg");
    cv::imwrite(outputImagePath, cpuImg);
    std::cout << "Saved annotated image to: " << outputImagePath << std::endl;

    return 0;
}
