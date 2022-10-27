import numpy as np
import open3d as o3d
from tqdm import tqdm

# Class for rendering point clouds from depth data

class CloudRenderer:
    def __init__(self, depth_data):
        # Initialize with depth data and process it
        self.depth_data = depth_data
        self.processed_data = self._process_data()
        self.point_cloud = self._generate_point_cloud()
    
    # TODO: Processes the depth data to generate depth points

    # TODO: Calculates normalization factor for depth data

    # TODO: Generate depth points from depth data using normalization factor

    # TODO: Generate point cloud from processed depth points
 
    # TODO: Displays point cloud

    # TODO: display voxel grid

 