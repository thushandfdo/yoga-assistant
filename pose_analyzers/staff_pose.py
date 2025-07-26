"""
Staff Pose Analyzer
Converts the Flask staff pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class StaffPoseAnalyzer(BasePoseAnalyzer):
    """Staff Pose (Dandasana) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Staff Pose specific angle ranges
        self.ideal_angles = {
            'right_elbow': (80, 120),     # Arms at sides
            'left_elbow': (80, 120),      # Arms at sides
            'right_shoulder': (60, 100),  # Shoulders relaxed
            'left_shoulder': (60, 100),   # Shoulders relaxed
            'right_hip': (150, 180),      # Hip extension
            'left_hip': (150, 180),       # Hip extension
            'right_knee': (160, 180),     # Legs straight
            'left_knee': (160, 180)       # Legs straight
        }
    
    def get_pose_name(self) -> str:
        return "Staff Pose (Dandasana)"
    
    def get_description(self) -> str:
        return "A foundational seated pose that improves posture and alignment"
    
    def get_difficulty(self) -> str:
        return "beginner"
    
    def get_benefits(self) -> List[str]:
        return [
            'Improves posture and alignment',
            'Strengthens back muscles',
            'Opens chest and shoulders',
            'Enhances body awareness',
            'Provides foundation for seated poses'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Sit with legs extended forward',
            'Place hands beside hips',
            'Lengthen spine and lift chest',
            'Flex feet and point toes up',
            'Keep legs straight and together',
            'Hold with steady breath'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Staff pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic Staff Pose analysis
        feedback = ["Staff Pose detected! Keep your spine straight and legs extended."]
        score = 76  # Placeholder score
        
        return self.create_success_response(
            pose_detected="Staff Pose",
            score=score,
            feedback=feedback
        ) 