"""
Lord of the Dance Pose Analyzer
Converts the Flask lord pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class LordPoseAnalyzer(BasePoseAnalyzer):
    """Lord of the Dance Pose (Natarajasana) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Lord of the Dance specific angle ranges
        self.ideal_angles = {
            'right_elbow': (80, 120),     # Arms extended
            'left_elbow': (80, 120),      # Arms extended
            'right_shoulder': (60, 100),  # Shoulder extension
            'left_shoulder': (60, 100),   # Shoulder extension
            'right_hip': (150, 180),      # Hip extension
            'left_hip': (150, 180),       # Hip extension
            'right_knee': (25, 45),       # Knee flexion (back leg)
            'left_knee': (25, 45)         # Knee flexion (back leg)
        }
    
    def get_pose_name(self) -> str:
        return "Lord of the Dance Pose (Natarajasana)"
    
    def get_description(self) -> str:
        return "A challenging balance pose that improves focus and leg strength"
    
    def get_difficulty(self) -> str:
        return "advanced"
    
    def get_benefits(self) -> List[str]:
        return [
            'Improves balance and concentration',
            'Strengthens standing leg and core',
            'Stretches hip flexors and shoulders',
            'Enhances focus and mental clarity',
            'Builds confidence and grace'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Stand on one leg',
            'Bend opposite knee behind',
            'Grasp foot with same hand',
            'Extend other arm forward',
            'Lift chest and gaze ahead',
            'Hold with steady breath'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Lord of the Dance pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic Lord of the Dance analysis
        feedback = ["Advanced pose detected! Focus on balance and gradual extension."]
        score = 68  # Placeholder score
        
        return self.create_success_response(
            pose_detected="Lord of the Dance Pose",
            score=score,
            feedback=feedback
        ) 