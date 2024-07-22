#pragma once

#include <argparse/argparse.hpp>

namespace argparseUtils {

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

// TODO: use cmake .in to define version based on `git describe` as in 
// https://gitlab.kitware.com/paraview/paraview/-/blob/master/CMakeLists.txt?ref_type=heads#L180
// https://gitlab.kitware.com/paraview/paraview/-/blob/master/CMake/paraview_plugin.h.in?ref_type=heads#L65
const std::string appVersion = "0.1";

}
