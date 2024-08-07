#!/usr/bin/env python3
"""
export_kwiver_pipe - A script to export a YAML configuration from a kwiver pipeline.
"""

from datetime import datetime
import argparse
from pathlib import Path
import re
import ast
import zipfile
import json

from matplotlib.cm import get_cmap


YAML_CONFIG_TEMPLATE = """%YAML:1.0
#converted from {pipe_path} {date}

model:
    type: {model_type}

image_pre_transforms:
    ConvertColor:
        model: "BGR"  #darknet specific which requires BGR input
    ResizeImg:
        size: [-1, -1]   # height width of net, read from input trt engine
        method: "{resize_method}"
    CastImg:
        dtype: "float"
        scale: true
    NormalizeImg:
        mean: [0.0, 0.0, 0.0]
        std: [1.0, 1.0, 1.0]

target_post_transforms:
    FilterBoxes:
        thresh: {threshold}  #block detector:darknet :thresh
    ConvertBox:
        src_fmt: "cxcywh"
    RescaleBox:
        offset: [0.0, 0.0]
        scale: [-1, -1]   # height width of net, read from input trt engine
    ResizeBox:
        size: [-1, -1]  # height width of image, read from input frame
        method:  "{resize_method}"
    NMS:
        max_overlap: {nms_max_overlap}
        nms_scale_factor: {nms_scale_factor}
        output_scale_factor: {nms_output_scale_factor}

labels:
{labels}

colors:
{colors}
"""


def get_parser():
    # parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=__doc__)
    parser.add_argument(
        "--pipe",
        help="Input pipeline path (.pipe)")
    parser.add_argument(
        "--kwiver-install-dir",
        default="",
        help="Path to the install directory of kwiver")

    return parser


def parse_model_type(pipe_path: Path):
    """Parse the model type from the pipeline file."""
    pattern = ".*?:detector:type *?([aA-zZ]*)\n"

    with open(pipe_path, "r") as f:
        for line in f:
            matches = re.match(pattern, line)
            if matches:
                model_type = matches[1]
            # Do not stop until end to get last model type

    return model_type


def parse_darknet_pipeline(pipe_path: Path):
    """
    Parse the darknet pre-processing options.

    Returns:
        resize_method: Resizing method (str)
        labels: Class labels (List[str])
        threshold: Box filter threshold (float)
    """
    labels = []
    threshold = 0.01
    resize_method = ""

    # labels
    yolo_labels_path = Path.joinpath(pipe_path.parent, "yolo.lbl")
    if not yolo_labels_path.exists():
        raise Exception(f"Cannot found darknet config file at {yolo_labels_path} !")
    with open(yolo_labels_path, "r") as f:
        labels = [line.strip() for line in f]
    # threshold and resize method
    with open(pipe_path, "r") as f:
        for line in f:
            # threshold
            pattern = ".*?:thresh *?([+-]?([0-9]*[.])?[0-9]+)\n"
            matches = re.match(pattern, line)
            if matches:
                threshold = float(matches[1])
            # Resize method
            pattern = ".*?:resize_option *?([aA-zZ]*)\n"
            matches = re.match(pattern, line)
            if matches:
                resize_method = matches[1]

    return resize_method, labels, threshold


def parse_netharn_pipeline(pipe_path: Path):
    """
    Parse the netharn pre-processing options.

    Returns:
        resize_method: Resizing method (str)
        labels: Class labels (List[str])
    """
    resize_method = ""
    labels = []

    # network shape and labels
    netharn_archive_name = "trained_detector.zip"
    netharn_cfg_name = "train_info.json"
    netharn_zip_path = Path.joinpath(pipe_path.parent, netharn_archive_name)
    with zipfile.ZipFile(netharn_zip_path, 'r') as zip_ref:
        all_files_in_zip = zip_ref.namelist()
        train_info_files = [file for file in all_files_in_zip if file.endswith(netharn_cfg_name)]
        if len(train_info_files) > 1:
            raise Exception(f"{netharn_zip_path} must contain one and only one {netharn_cfg_name}")
        with zip_ref.open(train_info_files[0]) as file:
            content = file.read()
            json_content = json.loads(content.decode('utf-8'))
            # labels
            labels = json_content["hyper"]["model"][1]["classes"]["idx_to_node"]
    # resize method
    with open(pipe_path, "r") as f:
        for line in f:
            # Resize method
            pattern = ".*?:mode *?([aA-zZ]*)\n"
            matches = re.match(pattern, line)
            if matches:
                resize_method = matches[1]
                # packages/kwiver/arrows/ocv/windowed_detector.cxx:l274
                if resize_method == "original_and_resized":
                    resize_method = "scale"

    return resize_method, labels


def parse_nms_pipeline(pipe_path: Path):
    """
    Parse the nms options.

    Returns:
        nms_max_overlap: Minimum IoU value to discard box (float)
        nms_scale_factor: (float)
        nms_output_scale_factor: (float)
    """
    nms_max_overlap = 0.50
    nms_scale_factor = 1.0
    nms_output_scale_factor = 1.0

    # threshold and resize method
    with open(pipe_path, "r") as f:
        for line in f:
            # nms max overlap
            pattern = ".*?:max_overlap *?([+-]?([0-9]*[.])?[0-9]+)\n"
            matches = re.match(pattern, line)
            if matches:
                nms_max_overlap = float(matches[1])
            # nms scale factor
            pattern = ".*?:nms_scale_factor *?([+-]?([0-9]*[.])?[0-9]+)\n"
            matches = re.match(pattern, line)
            if matches:
                nms_scale_factor = float(matches[1])
            # nms output scale factor
            pattern = ".*?:output_scale_factor *?([+-]?([0-9]*[.])?[0-9]+)\n"
            matches = re.match(pattern, line)
            if matches:
                nms_output_scale_factor = float(matches[1])

    return nms_max_overlap, nms_scale_factor, nms_output_scale_factor


def main(pipe_path: Path, kwiver_install_dir: Path = ""):

    if not pipe_path.exists():
        raise Exception(f"Cannot found pipeline file at {pipe_path}!")
    if kwiver_install_dir: 
        kwiver_bin = Path.joinpath(kwiver_install_dir, "bin", "kwiver")
        if not kwiver_bin.exists():
            raise Exception(f"Cannot found kwiver binary at {kwiver_bin}!")

    date = datetime.now().strftime("%Y-%d-%mT%H:%M:%S")
    resize_method = ""
    model_type = ""
    labels = []
    colors = []
    threshold = 0.01
    nms_max_overlap = 0.50
    nms_scale_factor = 1.0
    nms_output_scale_factor = 1.0

    # kwiver does not output pre-processing options in its graph !!!!
    # from networkx.drawing.nx_pydot import read_dot
    # dot_file = pipe_path.with_suffix(".dot").name
    # subprocess.call([f"{kwiver_bin}", "pipe-to-dot", "--pipe-file", f"{pipe_path}", "--name", "detector", "-o", f"{dot_file}"])
    # graph = read_dot(dot_file)
    # graphs = pydot.graph_from_dot_file(dot_file)
    # graph = graphs[0]

    model_type = parse_model_type(pipe_path)
    if model_type == "darknet":
        resize_method, labels, threshold = parse_darknet_pipeline(pipe_path)
    elif model_type == "netharn":
        resize_method, labels = parse_netharn_pipeline(pipe_path)
    nms_max_overlap, nms_scale_factor, nms_output_scale_factor = parse_nms_pipeline(pipe_path)
    # uniform sampling of colors on rainbow
    cmap = get_cmap("gist_rainbow")
    cmap_indexes = [float((val+1)/len(labels)) for val in range(len(labels))]
    colors = [cmap(index)[:3] for index in cmap_indexes]

    detector_config = Path("detector_config.yaml")
    with open(detector_config, "w") as f:
        labels_str = "\n".join([f"    - \"{lbl}\"" for lbl in labels])
        colors_str = "\n".join([f"    - {list(clr)}" for clr in colors])
        detector_config = YAML_CONFIG_TEMPLATE.format(
                pipe_path=pipe_path,
                date=date,
                model_type=model_type,
                resize_method=resize_method,
                threshold=threshold,
                nms_max_overlap=nms_max_overlap,
                nms_scale_factor=nms_scale_factor,
                nms_output_scale_factor=nms_output_scale_factor,
                labels=labels_str,
                colors=colors_str
            )
        f.write(detector_config)


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)
    pipe_path = Path("/home/ltetrel/Documents/data/models/darknet-yolo/keu/detector.pipe")
    pipe_path = Path("/home/ltetrel/Documents/data/models/deployment_cascadercnn_inference/detector.pipe")
    kwiver_install_dir = Path("/home/ltetrel/Documents/developments/ifremer/marine-beacon/VIAME/build/install")
    main(pipe_path, kwiver_install_dir)
