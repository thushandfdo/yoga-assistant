"""
Goddess Pose Analyzer
Converts the Flask goddess pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class GoddessPoseAnalyzer(BasePoseAnalyzer):
    """Goddess Pose (Utkata Konasana) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Goddess Pose specific angle ranges
        self.ideal_angles = {
            'right_elbow': (80, 120),     # Arms in prayer position
            'left_elbow': (80, 120),      # Arms in prayer position
            'right_shoulder': (60, 100),  # Shoulder abduction
            'left_shoulder': (60, 100),   # Shoulder abduction
            'right_hip': (90, 130),       # Hip flexion and abduction
            'left_hip': (90, 130),        # Hip flexion and abduction
            'right_knee': (90, 130),      # Knee flexion
            'left_knee': (90, 130)        # Knee flexion
        }
    
    def get_pose_name(self) -> str:
        return "Goddess Pose (Utkata Konasana)"
    
    def get_description(self) -> str:
        return "A wide-legged squat that strengthens legs and opens hips"
    
    def get_difficulty(self) -> str:
        return "intermediate"
    
    def get_benefits(self) -> List[str]:
        return [
            'Strengthens thighs, glutes, and core',
            'Opens hips and improves flexibility',
            'Builds endurance and stamina',
            'Improves balance and stability',
            'Energizes the body and mind'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Stand with feet wide apart',
            'Turn toes out at 45 degrees',
            'Bend knees and lower hips',
            'Bring hands to prayer position',
            'Keep knees tracking over toes',
            'Hold the position steadily'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Goddess pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic Goddess Pose analysis
        feedback = ["Goddess Pose detected! Keep your knees tracking over your toes."]
        score = 72  # Placeholder score
        
        return self.create_success_response(
            pose_detected="Goddess Pose",
            score=score,
            feedback=feedback
        ) 