"""
Side Plank Pose Analyzer
Converts the Flask side plank analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class SidePlankAnalyzer(BasePoseAnalyzer):
    """Side Plank Pose (Vasisthasana) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Side Plank specific angle ranges
        self.ideal_angles = {
            'right_elbow': (80, 120),     # Supporting arm
            'left_elbow': (80, 120),      # Supporting arm
            'right_shoulder': (60, 100),  # Shoulder stability
            'left_shoulder': (60, 100),   # Shoulder stability
            'right_hip': (150, 180),      # Hip extension
            'left_hip': (150, 180),       # Hip extension
            'right_knee': (160, 180),     # Legs straight
            'left_knee': (160, 180)       # Legs straight
        }
    
    def get_pose_name(self) -> str:
        return "Side Plank Pose (Vasisthasana)"
    
    def get_description(self) -> str:
        return "A core-strengthening pose that builds stability and strength"
    
    def get_difficulty(self) -> str:
        return "intermediate"
    
    def get_benefits(self) -> List[str]:
        return [
            'Strengthens core and obliques',
            'Builds shoulder and arm strength',
            'Improves balance and stability',
            'Enhances body awareness',
            'Develops side body strength'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Start in plank position',
            'Rotate to one side',
            'Stack feet and lift hips',
            'Extend top arm toward ceiling',
            'Keep body in straight line',
            'Hold with steady breath'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Side Plank pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic Side Plank analysis
        feedback = ["Side Plank detected! Keep your body in a straight line."]
        score = 71  # Placeholder score
        
        return self.create_success_response(
            pose_detected="Side Plank Pose",
            score=score,
            feedback=feedback
        ) 