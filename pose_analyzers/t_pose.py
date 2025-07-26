"""
T-Pose Analyzer
Converts the Flask t-pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class TPoseAnalyzer(BasePoseAnalyzer):
    """T-Pose (Tadasana with arms extended) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # T-Pose specific angle ranges
        self.ideal_angles = {
            'right_elbow': (160, 180),    # Arms straight
            'left_elbow': (160, 180),     # Arms straight
            'right_shoulder': (80, 120),  # Shoulder abduction
            'left_shoulder': (80, 120),   # Shoulder abduction
            'right_hip': (150, 180),      # Hip neutral
            'left_hip': (150, 180),       # Hip neutral
            'right_knee': (160, 180),     # Legs straight
            'left_knee': (160, 180)       # Legs straight
        }
    
    def get_pose_name(self) -> str:
        return "T-Pose (Tadasana with arms extended)"
    
    def get_description(self) -> str:
        return "A foundational standing pose with arms extended for calibration"
    
    def get_difficulty(self) -> str:
        return "beginner"
    
    def get_benefits(self) -> List[str]:
        return [
            'Improves posture and alignment',
            'Strengthens core and legs',
            'Opens chest and shoulders',
            'Enhances body awareness',
            'Provides calibration reference'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Stand with feet together',
            'Extend arms straight out to sides',
            'Keep spine straight and tall',
            'Engage core and lift chest',
            'Palms face forward',
            'Hold steady and breathe'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze T-Pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic T-Pose analysis
        feedback = ["T-Pose detected! Keep your arms straight and body aligned."]
        score = 78  # Placeholder score
        
        return self.create_success_response(
            pose_detected="T-Pose",
            score=score,
            feedback=feedback
        ) 