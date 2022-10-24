import os
import numpy as np
import matplotlib.pyplot as plt

import cv2 #OpenCV-python


class ImageProcessor:

    # Initialize w/ image data
    def __init__(self, data):
        self.data = self.__validate_data(data)
    

    # Validate image data
    def __validate_data(self, data):
        if data is None:
            raise ValueError("Data cannot be None")
        return data

    # Create ImageProcessor from ndarray
    @classmethod
    def from_ndarray(cls, ndarray):
        return cls(data=ndarray)

    # Return image dimensions
    @property
    def dimensions(self) -> tuple:
        return self.data.shape
    
    
    # Apply colormap to image data
    def apply_colormap(self, img_data) -> np.ndarray:
        img_normalized = img_data.astype(np.uint8)
        return cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
    
    # Show image (w/ optional colormap)
    def show(self, use_colormap:bool=False):
        result_img = self.__prepare_image_for_display(use_colormap)
        self.__display_image(result_img)
    