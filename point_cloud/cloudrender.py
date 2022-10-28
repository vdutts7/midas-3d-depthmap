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
    
    # Processes the depth data to generate depth points
    def _process_data(self) -> np.ndarray:
        normalization_factor = self._calculate_normalization_factor()
        depth_points = self._generate_depth_points(normalization_factor)
        print(f'Processed depth data from shape {self.depth_data.shape} to {depth_points.shape}')
        return depth_points 



    # Calculates normalization factor for depth data
    def _calculate_normalization_factor(self) -> float:
        normalize = lambda d: sum(d) / len(d)
        return normalize(self.depth_data.shape) / self.depth_data.max()




    # Generates depth points from depth data using normalization factor
    def _generate_depth_points(self, normalization_factor) -> np.ndarray:
        depth_points = []
        for y in tqdm(range(self.depth_data.shape[1])):
            for x in range(self.depth_data.shape[0]):
                adjusted_depth = self.depth_data[x][y] * normalization_factor
                point = np.array([x, y, adjusted_depth])
                depth_points.append(point)
        return np.array(depth_points)


    # Generates a point cloud from processed depth points
    def _generate_point_cloud(self):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.processed_data)
        return cloud
        
    # Displays the point cloud
    def display_cloud(self):
        self._display_geometry(self.point_cloud)
        
    # Displays a voxel grid created from the point cloud
    def display_voxel_grid(self):
        voxel_grid = self._create_voxel_grid()
        self._display_geometry(voxel_grid)


 