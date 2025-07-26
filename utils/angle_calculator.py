"""
Angle Calculation Utilities
Functions for calculating angles and distances in pose analysis
"""

import numpy as np
import math
from typing import List, Tuple


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calculate angle between three points using vector mathematics
    
    Args:
        a: First point [x, y]
        b: Vertex point [x, y] 
        c: Third point [x, y]
        
    Returns:
        Angle in degrees (0-180)
    """
    try:
        a = np.array(a)
        b = np.array(b)  # vertex
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clamp to valid range to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.arccos(cosine_angle) * 180.0 / np.pi
        
        return angle
        
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0.0


def calculate_angle_alternative(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Alternative angle calculation using arctangent method
    Used in some of the original pose analysis files
    
    Args:
        a: First point [x, y]
        b: Vertex point [x, y]
        c: Third point [x, y]
        
    Returns:
        Angle in degrees (0-180)
    """
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
        
    except Exception as e:
        print(f"Error calculating angle (alternative): {e}")
        return 0.0


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point [x, y]
        point2: Second point [x, y]
        
    Returns:
        Distance between points
    """
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return 0.0


def calculate_slope(point1: List[float], point2: List[float]) -> float:
    """
    Calculate slope between two points
    
    Args:
        point1: First point [x, y]
        point2: Second point [x, y]
        
    Returns:
        Slope value (or infinity for vertical lines)
    """
    try:
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        if abs(dx) < 1e-10:  # Vertical line
            return float('inf')
        
        return dy / dx
        
    except Exception as e:
        print(f"Error calculating slope: {e}")
        return 0.0


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to 0-360 degree range
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle (0-360)
    """
    try:
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
    except Exception:
        return 0.0


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the absolute difference between two angles
    Accounts for circular nature of angles
    
    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees
        
    Returns:
        Absolute difference in degrees (0-180)
    """
    try:
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    except Exception:
        return 0.0


def calculate_center_point(points: List[List[float]]) -> List[float]:
    """
    Calculate center point of multiple points
    
    Args:
        points: List of points [[x, y], [x, y], ...]
        
    Returns:
        Center point [x, y]
    """
    try:
        if not points:
            return [0.0, 0.0]
        
        x_sum = sum(point[0] for point in points)
        y_sum = sum(point[1] for point in points)
        
        center_x = x_sum / len(points)
        center_y = y_sum / len(points)
        
        return [center_x, center_y]
        
    except Exception as e:
        print(f"Error calculating center point: {e}")
        return [0.0, 0.0]


def is_point_between(point: List[float], line_start: List[float], line_end: List[float], tolerance: float = 0.1) -> bool:
    """
    Check if a point lies approximately on a line segment between two points
    
    Args:
        point: Point to check [x, y]
        line_start: Start of line segment [x, y]
        line_end: End of line segment [x, y]
        tolerance: Tolerance for "on line" check
        
    Returns:
        True if point is approximately on the line segment
    """
    try:
        # Calculate distances
        d1 = calculate_distance(line_start, point)
        d2 = calculate_distance(point, line_end)
        d_total = calculate_distance(line_start, line_end)
        
        # Check if sum of distances equals total distance (within tolerance)
        return abs(d1 + d2 - d_total) < tolerance
        
    except Exception:
        return False


def calculate_body_alignment(shoulder_left: List[float], shoulder_right: List[float], 
                           hip_left: List[float], hip_right: List[float]) -> dict:
    """
    Calculate body alignment metrics
    
    Args:
        shoulder_left: Left shoulder coordinates
        shoulder_right: Right shoulder coordinates  
        hip_left: Left hip coordinates
        hip_right: Right hip coordinates
        
    Returns:
        Dictionary with alignment metrics
    """
    try:
        # Calculate shoulder and hip center points
        shoulder_center = calculate_center_point([shoulder_left, shoulder_right])
        hip_center = calculate_center_point([hip_left, hip_right])
        
        # Calculate shoulder levelness
        shoulder_level_diff = abs(shoulder_left[1] - shoulder_right[1])
        
        # Calculate hip levelness
        hip_level_diff = abs(hip_left[1] - hip_right[1])
        
        # Calculate torso alignment (vertical deviation)
        torso_deviation = abs(shoulder_center[0] - hip_center[0])
        
        # Calculate torso angle
        torso_angle = calculate_angle([shoulder_center[0], shoulder_center[1] - 0.1], 
                                    shoulder_center, hip_center)
        
        return {
            'shoulder_level_diff': shoulder_level_diff,
            'hip_level_diff': hip_level_diff,
            'torso_deviation': torso_deviation,
            'torso_angle': torso_angle,
            'alignment_score': max(0, 1.0 - (shoulder_level_diff + hip_level_diff + torso_deviation))
        }
        
    except Exception as e:
        print(f"Error calculating body alignment: {e}")
        return {
            'shoulder_level_diff': 0,
            'hip_level_diff': 0,
            'torso_deviation': 0,
            'torso_angle': 90,
            'alignment_score': 0
        } 