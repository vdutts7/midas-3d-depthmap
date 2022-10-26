import os
import argparse
from image_processing import ImageProcessor

from depth_estimation import DepthMapper


# Sets up the command line argument parser
def setup_arg_parser() -> argparse.ArgumentParser:
    cmd_parser = argparse.ArgumentParser(description="Depth Mapping Utility")
    add_arguments_to_parser(cmd_parser)
    return cmd_parser

# Adds required arguments to the parser
def add_arguments_to_parser(parser):
    parser.add_argument("--input_img", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--accuracy_level", "-l", type=int, required=False, default=2, help="Accuracy level for depth estimation")


# Processes the image and returns a depth image
def process_image(image_path, accuracy_level):
    depth_estimator = DepthMapper(accuracy_level)
    img_instance = ImageProcessor.from_file(image_path)
    estimated_depth = depth_estimator.estimate_depth(img_instance.image)
    return ImageProcessor.from_array(estimated_depth)


# Main execution function (something...)
