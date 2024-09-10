# TensorRT C++ API

A minimalist object-detection API for [ONNX](https://onnx.ai/) /[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) models.

# Requirements

- CUDA `11.4`
- TensorRT `8.5`
- OpenCV `4.8` ([compiled with cuda](scripts/install_opencv4.8.0_Jetson.sh))

# Installation

```sh
git clone https://gitlab.ifremer.fr/lt330b2/tensorrt-cpp-api.git
cd tensorrt-cpp-api
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

# Usage

To use this project, you will need an [ONNX](https://onnx.ai/) model and a [detector config file](#detector-config).

The first time you instanciate a detector, a `TensorRT` "engine" file will be generated (`TensorRT` "compile" the instructions for the GPU), so the first run usually takes longer.

> **Warning**:
> The program currently check if the engine has already been generated based **on the ONNX filename**.
> If you generate a new model, make sure to rename the ONNX with another name or no engine will be generated.

## CLI

```
Usage: ./build/run_prediction_image [--help] [--version] --model VAR --cfg VAR --image VAR

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  --model        Path to the model (.onnx or .trt) [required]
  --cfg          Path to the inference config file (.yaml) [required]
  --image        Path to an image (check supported formats here: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#imread [required]
```

## C++ API

```c++
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include "detector.h"

// define paths
std::filesystem::path modelPath = ...
std::filesystem::path cfgPath = ...
std::filesystem::path imagePath = ...

// Read image and put to GPU
cv::Mat cpuImg = cv::imread(imagePath);
cv::cuda::GpuMat gpuImg;
gpuImg.upload(cpuImg);

// Perform prediction
Detector detector(modelPath, cfgPath);
std::vector<BoundingBox> detections = detector.mPredict(gpuImg);
```

## Detector config

The `Detector` class parse its definition from a YAML file.

It is decomposed into three mandatory sections: the **model definition**, the different **transforms** applied to the data and the **labels description**.

An example of config used by the tests for darknet backend is given [here](https://data.kitware.com/#item/66bf4cb2af422925a420eaf4).

```yaml
%YAML:1.0

model:
  ...

image_pre_transforms:
  ...
target_post_transforms:
  ...

labels:
  ...
colors:
  ...
```

### Model definition

This is where we inform the detector what type of machine learning model is used and was serialized by TensorRT.

Two parameters are required:
- `type`: defines the type of the model, usually standard pytorch detector architecture. Check [here](src/configParser.cpp:l10) to see a list of available model types.
- `backend`: what library was used when exporting to ONNX. This is important since this informs us about the underlying data formats inside the model.

```yaml
model:
    type: yolov4-csp-s-mish
    backend: darknet
```

### Transforms

This is the key section that defines all functions that are used by the `Detector`. There are two types of transformations:
- `image_pre_transforms`: This is the list of pre-processing functions applied on the input image. Those functions are wrapped in CUDA for maximum performance.
- `target_post_transforms`: The list of post-processing functions applied on the input bounding box. Those operations will run on the CPU.

For an exhaustive list of all image and box transforms, you can check [here](src/transforms.h:l86). Each of those transforms have different options depending on the operation. For example for `ConvertColorImg` there is one parameter `model` that defines what color model you want to convert to (`RGB`, `BGR`, or `GRAY`). 

```yaml
ConvertColorImg:
    model: "BGR"
```

All transforms will be chained in order so be carefull!

For some parameters, it is impossible to know in advance it value. Typically we cannot know in advance the target image size, because it depends on the input image! In that case, simply put `null` in the corresponding parameter.

```yaml
ResizeBBox:
    size: [null, null]  # height width of image, read from input frame
    method:  "maintain_ar"
```

## Labels description

The last section is the labels description mostly used when drawing the predictions to an image:
- `labels`: defines the different class names for the predictions, where the order is important. If the model outputs `0` then the corresponding class name would be the first in the list.
- `colors`: the different colors for each class in floating-point format [`R`, `G`, `B`]. Should keep the same order as for `labels`.

```yaml
labels:
    - "Scallop"
    - "cucumaria_frondosa"

colors:
    - [0.36036036036036034, 1.0, 0.0]
    - [0.0, 0.5615942028985503, 1.0]
```

# Testing

```
cd build
ctest
```

By default, all test data will be automatically downloaded from kitware girder in [this collection](https://data.kitware.com/#collection/66bf4b49af422925a420eada).

If you want to customize and use your own data for testing (not recommended), re-configure cmake with `-DDATA_TESTING_DIR=path/to/data/tests` and re-build.

# Benchmarking

Desktop (linux 5.15): CPU Intel i7-11800H @ 2.3GHz / GPU Nvidia RTX 3050 Ti Laptop.

|        Model         |   yolov4-csp-s-mish  |
|----------------------|----------------------|
| Pre-processing (ms)  | 0.79578 +/- 0.05134  |
| Inference (ms)       | 4.72340 +/- 0.05599  |
| Post-processing (ms) | 4.87697 +/- 0.06488  |

Jetson Xavier NX (tegra 5.10): CPU ARMv8 8-core @ 1.420GHz / GPU Nvidia Volta 512-core (64 Tensor cores).

|        Model         |   yolov4-csp-s-mish  |
|----------------------|----------------------|
| Pre-processing (ms)  | 3.79386  +/- 0.23677 |
| Inference (ms)       | 24.32557 +/- 1.99187 |
| Post-processing (ms) | 0.76734  +/- 0.12630 |

## Enable benchmark

Use the following cmake command:
```sh
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_BENCHMARK=ON ..
```
You can then run  the executable `run_prediction_image` and you should see the benchmark outpout.

> **Warning**:
> Do not use in production environment, since this impacts the processing time.

# Usefull links

TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/index.html
https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/index.html

# Jetson performance mode

Jetson devices allows you to switch between different modes.
By default the GPU is really slow so we recommend you to boost it.
```
sudo /usr/sbin/nvpmodel -m <mode-id>
```
You can find the different mode-id for the jetson NX [here](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#).

# Acknowledgement

https://github.com/cyrusbehr/tensorrt-cpp-api
