import torch
from time import perf_counter

class DepthMapper:
    def __init__(self, accuracy:int=2):
        # Initialize model, device, and transforms
        self.setup_model(accuracy)
        self.setup_device()
        self.setup_transforms(accuracy)

        
    def setup_model(self, accuracy):
        # Define model types and load the selected model
        depth_models = {
            1: "MiDaS_small",
            2: "DPT_Hybrid",
            3: "DPT_Large"
        }
        self.depth_model = torch.hub.load("intel-isl/MiDaS", depth_models[accuracy])
    
    def setup_device(self):
        # Set computation device based on CUDA availability
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(compute_device)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        print(f'Depth model ready, using {self.device}')
    
    def setup_transforms(self, accuracy):
        # Load and set image transformations based on model type
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.img_transform = transforms.dpt_transform if accuracy > 1 else transforms.small_transform
    
    
    #...............................
    
    
    def estimate_depth(self, img):
        # Estimate depth from an image
        img_prepared = self.prepare_image(img)
        depth_pred = self.run_model(img_prepared, img.shape[:2])
        return depth_pred.cpu().numpy()
    
    def prepare_image(self, img):
        # Prepare image for depth estimation
        return self.img_transform(img).to(self.device)
    