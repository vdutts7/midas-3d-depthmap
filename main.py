import os
import argparse
# TODO: depth estimate func 

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

# Main execution function (something...)
