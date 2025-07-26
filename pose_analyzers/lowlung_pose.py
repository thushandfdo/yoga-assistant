"""
Low Lunge Pose Analyzer
Converts the Flask lowlung pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class LowlungPoseAnalyzer(BasePoseAnalyzer):
    """Low Lunge Pose (Anjaneyasana) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Low Lunge specific angle ranges
        self.ideal_angles = {
            'right_elbow': (80, 120),     # Arms extended
            'left_elbow': (80, 120),      # Arms extended
            'right_shoulder': (60, 100),  # Shoulder extension
            'left_shoulder': (60, 100),   # Shoulder extension
            'right_hip': (90, 130),       # Hip flexion (front leg)
            'left_hip': (90, 130),        # Hip flexion (front leg)
            'right_knee': (90, 130),      # Knee flexion (front leg)
            'left_knee': (90, 130)        # Knee flexion (front leg)
        }
    
    def get_pose_name(self) -> str:
        return "Low Lunge Pose (Anjaneyasana)"
    
    def get_description(self) -> str:
        return "A hip-opening lunge that strengthens legs and improves flexibility"
    
    def get_difficulty(self) -> str:
        return "beginner"
    
    def get_benefits(self) -> List[str]:
        return [
            'Opens hip flexors and groin',
            'Strengthens quadriceps and glutes',
            'Improves balance and stability',
            'Stretches back leg muscles',
            'Enhances hip mobility'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Step one foot forward',
            'Bend front knee to 90 degrees',
            'Lower back knee to ground',
            'Lift chest and extend arms',
            'Keep front knee over ankle',
            'Hold and breathe deeply'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Low Lunge pose and provide feedback"""
        
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response('No pose detected.')
        
        landmarks = results.pose_landmarks.landmark
        joint_coords = self.extract_landmarks(landmarks)
        
        if not joint_coords:
            return self.create_error_response('Could not extract pose landmarks properly.')
        
        # Basic Low Lunge analysis
        feedback = ["Low Lunge detected! Keep your front knee over your ankle."]
        score = 74  # Placeholder score
        
        return self.create_success_response(
            pose_detected="Low Lunge Pose",
            score=score,
            feedback=feedback
        ) 