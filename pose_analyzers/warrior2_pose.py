"""
Warrior 2 Pose Analyzer
Converts the Flask warrior2 pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class Warrior2PoseAnalyzer(BasePoseAnalyzer):
    """Warrior 2 Pose (Virabhadrasana II) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Warrior 2 specific angle ranges
        self.ideal_angles = {
            'right_elbow': (80, 120),     # Arms extended
            'left_elbow': (80, 120),      # Arms extended
            'right_shoulder': (80, 120),  # Shoulder abduction
            'left_shoulder': (80, 120),   # Shoulder abduction
            'right_hip': (90, 130),       # Hip flexion (front leg)
            'left_hip': (90, 130),        # Hip flexion (front leg)
            'right_knee': (90, 130),      # Knee flexion (front leg)
            'left_knee': (160, 180)       # Back leg straight
        }
    
    def get_pose_name(self) -> str:
        return "Warrior 2 Pose (Virabhadrasana II)"
    
    def get_description(self) -> str:
        return "A powerful standing pose that builds strength and stability"
    
    def get_difficulty(self) -> str:
        return "intermediate"
    
    def get_benefits(self) -> List[str]:
        return [
            'Strengthens legs and core',
            'Improves balance and stability',
            'Opens hips and chest',
            'Builds endurance and focus',
            'Enhances body awareness'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Step feet wide apart',
            'Turn front foot out 90 degrees',
            'Bend front knee to 90 degrees',
            'Extend arms parallel to ground',
            'Keep back leg straight',
            'Gaze over front hand'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Warrior 2 pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic Warrior 2 analysis
        feedback = ["Warrior 2 detected! Keep your front knee over your ankle."]
        score = 73  # Placeholder score
        
        return self.create_success_response(
            pose_detected="Warrior 2 Pose",
            score=score,
            feedback=feedback
        ) 