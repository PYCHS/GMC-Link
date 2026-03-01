"""
KITTI Calibration Parser for Metric 3D Projection
==================================================
Extracts camera intrinsic parameters from KITTI calib.txt files.
Enables conversion from pixel coordinates to metric world coordinates.
"""

import numpy as np
from typing import Optional, Tuple
import os


class CameraCalibration:
    """
    Parses and stores KITTI camera calibration parameters.
    
    The P2 projection matrix format in KITTI calib.txt:
    P2: fx  0  cx  0
         0 fy  cy  0
         0  0   1  0
    
    Where:
    - fx, fy: Focal lengths in pixels
    - cx, cy: Principal point (optical center) in pixels
    """
    
    def __init__(self, calib_path: Optional[str] = None):
        """
        Initialize calibration from file or use KITTI defaults.
        
        Args:
            calib_path: Path to calib.txt file. If None, uses KITTI defaults.
        """
        if calib_path and os.path.exists(calib_path):
            self.load_from_file(calib_path)
        else:
            # KITTI default calibration parameters (from sequence 0000)
            self.fx = 718.856  # Focal length x (pixels)
            self.fy = 718.856  # Focal length y (pixels)
            self.cx = 607.1928  # Principal point x (pixels)
            self.cy = 185.2157  # Principal point y (pixels)
            
    def load_from_file(self, calib_path: str) -> None:
        """
        Parse calibration file and extract camera parameters.
        
        Args:
            calib_path: Path to KITTI calib.txt file
        """
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith('P2:'):
                # Parse P2 projection matrix: P2: fx 0 cx 0 0 fy cy 0 0 0 1 0
                values = line.split()[1:]  # Skip 'P2:' label
                values = [float(v) for v in values]
                
                # Extract intrinsic parameters
                self.fx = values[0]   # P[0, 0]
                self.fy = values[5]   # P[1, 1]
                self.cx = values[2]   # P[0, 2]
                self.cy = values[6]   # P[1, 2]
                break
    
    def pixel_to_metric_velocity(
        self, 
        dx_pixel: float, 
        dy_pixel: float, 
        depth: float, 
        time_delta: float = 1.0
    ) -> Tuple[float, float]:
        """
        Convert pixel displacement to metric world velocity (m/s).
        
        Formula:
            V_world_x = (Δx_pixel · Z) / (f_x · Δt)
            V_world_y = (Δy_pixel · Z) / (f_y · Δt)
        
        This resolves the parallax problem: stationary objects at different
        depths will have zero world velocity even if they have pixel motion
        due to camera movement.
        
        Args:
            dx_pixel: Pixel displacement in x direction
            dy_pixel: Pixel displacement in y direction
            depth: Estimated depth (Z) in meters
            time_delta: Time between frames (seconds), default=1.0
            
        Returns:
            (v_x, v_y): Velocity in world frame (meters/second)
        """
        # Avoid division by zero
        if depth <= 0:
            depth = 1.0  # Fallback to 1 meter
            
        v_x = (dx_pixel * depth) / (self.fx * time_delta)
        v_y = (dy_pixel * depth) / (self.fy * time_delta)
        
        return v_x, v_y
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Get camera intrinsic matrix K.
        
        Returns:
            K: 3x3 intrinsic matrix
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    def __repr__(self) -> str:
        return (f"CameraCalibration(fx={self.fx:.3f}, fy={self.fy:.3f}, "
                f"cx={self.cx:.3f}, cy={self.cy:.3f})")


def estimate_depth_from_bbox(
    bbox_height: float, 
    image_height: float = 375.0,
    avg_object_height: float = 1.5
) -> float:
    """
    Estimate object depth from bounding box height (simple heuristic).
    
    This is a placeholder for more sophisticated depth estimation.
    In production, you would use:
    - Monocular depth estimation networks (e.g., MiDaS, DPT)
    - Stereo depth maps (if available)
    - LiDAR data (if available)
    
    Formula:
        Z ≈ (f_y · H_real) / h_pixel
    
    Args:
        bbox_height: Height of bounding box in pixels
        image_height: Full image height in pixels
        avg_object_height: Average real-world object height (meters)
        
    Returns:
        Estimated depth in meters
    """
    if bbox_height <= 0:
        return 10.0  # Default: 10 meters
    
    # Simple inverse relationship: larger bbox = closer object
    # This is a very rough approximation
    focal_length = 718.856  # KITTI default
    depth = (focal_length * avg_object_height) / bbox_height
    
    # Clamp to reasonable range for driving scenarios
    depth = np.clip(depth, 2.0, 100.0)  # 2m to 100m
    
    return float(depth)
