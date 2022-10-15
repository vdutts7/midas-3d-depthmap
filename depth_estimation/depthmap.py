import torch
from time import perf_counter

class DepthMapper:
    def __init__(self, accuracy:int=2):
        # Initialize model, device, and transforms
        self.setup_model(accuracy)
        self.setup_device()
        self.setup_transforms(accuracy)