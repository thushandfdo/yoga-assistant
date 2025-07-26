"""
Base Pose Analyzer Class
Provides common functionality for all yoga pose analyzers
"""

from abc import ABC, abstractmethod
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class BasePoseAnalyzer(ABC):
    """
    Abstract base class for all yoga pose analyzers
    Provides common functionality and enforces interface consistency
    """
    
    def __init__(self):
        self.pose_name = self.get_pose_name()
        self.description = self.get_description()
        self.difficulty = self.get_difficulty()
        
        # Common MediaPipe settings
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.7
    
    @abstractmethod
    def get_pose_name(self) -> str:
        """Return the name of the pose"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return description of the pose"""
        pass
    
    @abstractmethod
    def get_difficulty(self) -> str:
        """Return difficulty level (beginner, intermediate, advanced)"""
        pass
    
    @abstractmethod
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """
        Analyze the pose in the given image
        
        Args:
            image: Input image as numpy array
            skill_level: User's skill level
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def get_pose_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the pose
        Default implementation - can be overridden by subclasses
        """
        return {
            "pose_name": self.get_pose_name(),
            "description": self.get_description(),
            "difficulty": self.get_difficulty(),
            "benefits": self.get_benefits(),
            "instructions": self.get_instructions(),
            "common_mistakes": self.get_common_mistakes(),
            "modifications": self.get_modifications()
        }
    
    def get_benefits(self) -> List[str]:
        """Return list of pose benefits - can be overridden"""
        return [
            "Improves balance and stability",
            "Strengthens core muscles",
            "Enhances flexibility",
            "Promotes mindfulness and focus"
        ]
    
    def get_instructions(self) -> List[str]:
        """Return step-by-step instructions - can be overridden"""
        return [
            "Position yourself in clear view of the camera",
            "Follow the pose guidance",
            "Hold the position steadily",
            "Breathe deeply and maintain focus"
        ]
    
    def get_common_mistakes(self) -> List[str]:
        """Return common mistakes - can be overridden"""
        return [
            "Poor alignment",
            "Holding breath",
            "Rushing the pose",
            "Ignoring body limitations"
        ]
    
    def get_modifications(self) -> List[str]:
        """Return pose modifications - can be overridden"""
        return [
            "Use props for support if needed",
            "Reduce intensity for beginners", 
            "Hold for shorter duration initially",
            "Focus on proper alignment over depth"
        ]
    
    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """
        Calculate angle between three points using vector mathematics
        
        Args:
            a, b, c: Points as [x, y] coordinates
            
        Returns:
            Angle in degrees
        """
        try:
            a = np.array(a)
            b = np.array(b)  # vertex
            c = np.array(c)
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """
        Calculate normalized distance between two points
        
        Args:
            point1, point2: Points as [x, y] coordinates
            
        Returns:
            Euclidean distance
        """
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except Exception:
            return 0.0
    
    def extract_landmarks(self, landmarks) -> Optional[Dict[str, List[float]]]:
        """
        Extract key body landmarks from MediaPipe results
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of landmark coordinates or None if extraction fails
        """
        try:
            coords = {
                'nose': [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y],
                'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
                'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
                'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
                'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            }
            return coords
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def detect_pose_landmarks(self, image: np.ndarray) -> Optional[Any]:
        """
        Detect pose landmarks using MediaPipe
        
        Args:
            image: Input image as numpy array
            
        Returns:
            MediaPipe pose results or None if detection fails
        """
        try:
            with mp_pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            ) as pose:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process the image
                results = pose.process(image_rgb)
                
                return results
        except Exception as e:
            print(f"Error detecting pose landmarks: {e}")
            return None
    
    def create_error_response(self, message: str, additional_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create standardized error response
        
        Args:
            message: Error message
            additional_info: Optional additional information
            
        Returns:
            Standardized error response dictionary
        """
        response = {
            'success': False,
            'message': message,
            'pose_detected': 'None',
            'score': 0,
            'feedback': [message],
            'error': True
        }
        
        if additional_info:
            response.update(additional_info)
            
        return response
    
    def create_success_response(
        self,
        pose_detected: str,
        score: int,
        feedback: List[str],
        angles: Optional[Dict[str, int]] = None,
        additional_metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create standardized success response
        
        Args:
            pose_detected: Name of detected pose
            score: Pose quality score (0-100)
            feedback: List of feedback messages
            angles: Joint angles dictionary
            additional_metrics: Optional additional measurements
            
        Returns:
            Standardized success response dictionary
        """
        response = {
            'success': True,
            'pose_detected': pose_detected,
            'score': score,
            'feedback': feedback,
            'message': f"Score: {score}% - {pose_detected}",
            'error': False
        }
        
        if angles:
            response['angles'] = angles
            
        if additional_metrics:
            response['metrics'] = additional_metrics
            
        return response 