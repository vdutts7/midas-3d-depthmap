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


    # Load image from disk --> convert to RGB
    @classmethod
    def load_from_disk(cls, path):
        img_rgb = cls.__read_and_convert_image(path)
        return cls(data=img_rgb)
    
    # Read image from path --> convert to RGB
    @staticmethod
    def __read_and_convert_image(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Image \"{path}\" not found.')
        
        img_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    

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
    
    # Prepare image for display --> apply colormap (if needed)
    def __prepare_image_for_display(self, use_colormap):
        apply_colormap = lambda img: cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return apply_colormap(self.data) if use_colormap else self.data
    
    # Display image (via matplotlib)
    @staticmethod
    def __display_image(image):
        print("Rendering image...")
        plt.axis("off")
        plt.imshow(image)