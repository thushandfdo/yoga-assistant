"""
Downward Dog Pose Analyzer
Converts the Flask downdog pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class DowndogPoseAnalyzer(BasePoseAnalyzer):
    """Downward Dog Pose (Adho Mukha Svanasana) analyzer"""
    
    def __init__(self):
        super().__init__()
        
        # Downward Dog specific angle ranges
        self.ideal_angles = {
            'right_elbow': (160, 180),    # Arms straight
            'left_elbow': (160, 180),     # Arms straight
            'right_shoulder': (80, 120),  # Shoulder flexion
            'left_shoulder': (80, 120),   # Shoulder flexion
            'right_hip': (150, 180),      # Hip extension
            'left_hip': (150, 180),       # Hip extension
            'right_knee': (160, 180),     # Legs straight
            'left_knee': (160, 180)       # Legs straight
        }
    
    def get_pose_name(self) -> str:
        return "Downward Dog Pose (Adho Mukha Svanasana)"
    
    def get_description(self) -> str:
        return "An inversion pose that strengthens and stretches the entire body"
    
    def get_difficulty(self) -> str:
        return "beginner"
    
    def get_benefits(self) -> List[str]:
        return [
            'Strengthens arms, shoulders, and legs',
            'Stretches hamstrings, calves, and spine',
            'Improves circulation and energy flow',
            'Calms the mind and relieves stress',
            'Builds core strength and stability'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Start on hands and knees',
            'Lift hips up and back',
            'Straighten arms and legs',
            'Form an inverted V-shape',
            'Press through hands and feet',
            'Keep head between arms'
        ]
    
    def get_common_mistakes(self) -> List[str]:
        return [
            'Rounding the back',
            'Bending knees too much',
            'Not pressing through hands',
            'Collapsing in shoulders',
            'Looking up instead of down'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Downward Dog pose and provide feedback"""
        
        # Detect pose landmarks
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response(
                'No pose detected. Please ensure you are visible in the frame.'
            )
        
        landmarks = results.pose_landmarks.landmark
        
        try:
            # Extract joint coordinates
            joint_coords = self.extract_landmarks(landmarks)
            
            if not joint_coords:
                return self.create_error_response(
                    'Could not extract pose landmarks properly.'
                )
            
            # Calculate angles
            angles = self._calculate_pose_angles(joint_coords)
            
            if not angles or len(angles) != 8:
                return self.create_error_response(
                    'Could not calculate pose angles properly.'
                )
            
            # Analyze mistakes and get feedback
            mistakes = self._analyze_pose_mistakes(angles, joint_coords)
            
            # Get pose classification
            is_downdog_pose, confidence = self._classify_downdog_pose(angles)
            
            final_score = int(confidence * 100)
            
            # Determine pose label
            pose_label = "Downward Dog Pose" if is_downdog_pose else "Working towards Downward Dog"
            
            # Create angle dictionary
            angle_dict = {
                'right_elbow': int(angles[0]),
                'left_elbow': int(angles[1]),
                'right_shoulder': int(angles[2]),
                'left_shoulder': int(angles[3]),
                'right_hip': int(angles[4]),
                'left_hip': int(angles[5]),
                'right_knee': int(angles[6]),
                'left_knee': int(angles[7])
            }
            
            return self.create_success_response(
                pose_detected=pose_label,
                score=final_score,
                feedback=mistakes,
                angles=angle_dict
            )
                    
        except Exception as e:
            return self.create_error_response(
                f'Error analyzing pose: {str(e)}'
            )
    
    def _calculate_pose_angles(self, joint_coords: Dict[str, List[float]]) -> List[float]:
        """Calculate all relevant angles for Downward Dog analysis"""
        try:
            angles = []
            
            # Right arm elbow angle (shoulder-elbow-wrist)
            angle1 = self.calculate_angle(joint_coords['right_shoulder'], 
                                        joint_coords['right_elbow'], 
                                        joint_coords['right_wrist'])
            angles.append(angle1)
            
            # Left arm elbow angle (shoulder-elbow-wrist)
            angle2 = self.calculate_angle(joint_coords['left_shoulder'], 
                                        joint_coords['left_elbow'], 
                                        joint_coords['left_wrist'])
            angles.append(angle2)
            
            # Right shoulder angle (elbow-shoulder-hip)
            angle3 = self.calculate_angle(joint_coords['right_elbow'], 
                                        joint_coords['right_shoulder'], 
                                        joint_coords['right_hip'])
            angles.append(angle3)
            
            # Left shoulder angle (elbow-shoulder-hip)
            angle4 = self.calculate_angle(joint_coords['left_elbow'], 
                                        joint_coords['left_shoulder'], 
                                        joint_coords['left_hip'])
            angles.append(angle4)
            
            # Right hip angle (shoulder-hip-knee)
            angle5 = self.calculate_angle(joint_coords['right_shoulder'], 
                                        joint_coords['right_hip'], 
                                        joint_coords['right_knee'])
            angles.append(angle5)
            
            # Left hip angle (shoulder-hip-knee)
            angle6 = self.calculate_angle(joint_coords['left_shoulder'], 
                                        joint_coords['left_hip'], 
                                        joint_coords['left_knee'])
            angles.append(angle6)
            
            # Right knee angle (hip-knee-ankle)
            angle7 = self.calculate_angle(joint_coords['right_hip'], 
                                        joint_coords['right_knee'], 
                                        joint_coords['right_ankle'])
            angles.append(angle7)
            
            # Left knee angle (hip-knee-ankle)
            angle8 = self.calculate_angle(joint_coords['left_hip'], 
                                        joint_coords['left_knee'], 
                                        joint_coords['left_ankle'])
            angles.append(angle8)
            
            return angles
            
        except Exception as e:
            print(f"Error calculating pose angles: {e}")
            return []
    
    def _classify_downdog_pose(self, angles: List[float]) -> tuple:
        """Classify if the detected pose is a proper Downward Dog"""
        if not angles or len(angles) != 8:
            return False, 0.0
        
        try:
            # Check each angle against ideal ranges
            angle_scores = []
            angle_names = ['right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                          'right_hip', 'left_hip', 'right_knee', 'left_knee']
            
            for i, (angle_name, angle_value) in enumerate(zip(angle_names, angles)):
                ideal_min, ideal_max = self.ideal_angles[angle_name]
                
                if ideal_min <= angle_value <= ideal_max:
                    score = 1.0
                else:
                    # Calculate how far off the angle is
                    if angle_value < ideal_min:
                        diff = ideal_min - angle_value
                    else:
                        diff = angle_value - ideal_max
                    
                    # Score decreases exponentially with distance from ideal range
                    score = max(0.0, 1.0 - (diff / 30.0))  # 30 degrees tolerance
                
                angle_scores.append(score)
            
            # Calculate weighted overall score
            arm_score = (angle_scores[0] + angle_scores[1] + angle_scores[2] + angle_scores[3]) / 4
            leg_score = (angle_scores[4] + angle_scores[5] + angle_scores[6] + angle_scores[7]) / 4
            
            overall_score = (arm_score * 0.5 + leg_score * 0.5)  # Equal weight for arms and legs
            
            is_downdog_pose = overall_score >= 0.7  # 70% threshold
            
            return is_downdog_pose, overall_score
            
        except Exception as e:
            print(f"Error in pose classification: {e}")
            return False, 0.0
    
    def _analyze_pose_mistakes(self, angles: List[float], joint_coords: Dict[str, List[float]]) -> List[str]:
        """Analyze specific mistakes in Downward Dog and provide corrections"""
        mistakes = []
        
        if not angles or len(angles) != 8:
            mistakes.append("Cannot detect pose properly. Please ensure you're fully visible.")
            return mistakes
        
        try:
            angle_names = ['right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                          'right_hip', 'left_hip', 'right_knee', 'left_knee']
            
            # Check each angle
            for i, (angle_name, current_angle) in enumerate(zip(angle_names, angles)):
                ideal_min, ideal_max = self.ideal_angles[angle_name]
                
                if current_angle < ideal_min - 10:  # Significant deviation
                    if angle_name == 'right_elbow':
                        mistakes.append(f"Right Arm: Straighten your right arm more (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_elbow':
                        mistakes.append(f"Left Arm: Straighten your left arm more (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_shoulder':
                        mistakes.append(f"Right Shoulder: Press through your right shoulder (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_shoulder':
                        mistakes.append(f"Left Shoulder: Press through your left shoulder (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_hip':
                        mistakes.append(f"Right Hip: Lift your hips higher (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_hip':
                        mistakes.append(f"Left Hip: Lift your hips higher (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_knee':
                        mistakes.append(f"Right Leg: Straighten your right leg more (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_knee':
                        mistakes.append(f"Left Leg: Straighten your left leg more (current: {current_angle:.1f}°)")
                        
                elif current_angle > ideal_max + 10:  # Significant deviation
                    if angle_name == 'right_elbow':
                        mistakes.append(f"Right Arm: Don't lock your right elbow (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_elbow':
                        mistakes.append(f"Left Arm: Don't lock your left elbow (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_knee':
                        mistakes.append(f"Right Leg: Don't hyperextend your right knee (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_knee':
                        mistakes.append(f"Left Leg: Don't hyperextend your left knee (current: {current_angle:.1f}°)")
            
            # Additional postural checks
            if joint_coords:
                # Check for inverted V-shape
                shoulder_center_y = (joint_coords['left_shoulder'][1] + joint_coords['right_shoulder'][1]) / 2
                hip_center_y = (joint_coords['left_hip'][1] + joint_coords['right_hip'][1]) / 2
                
                if hip_center_y > shoulder_center_y:
                    mistakes.append("Hips: Lift your hips higher to form an inverted V-shape")
                
                # Check spine alignment
                shoulder_center_x = (joint_coords['left_shoulder'][0] + joint_coords['right_shoulder'][0]) / 2
                hip_center_x = (joint_coords['left_hip'][0] + joint_coords['right_hip'][0]) / 2
                
                if abs(shoulder_center_x - hip_center_x) > 0.1:
                    mistakes.append("Spine: Keep your spine straight and aligned")
            
            if not mistakes:
                mistakes.append("Perfect Downward Dog! Excellent form!")
            
            return mistakes
            
        except Exception as e:
            print(f"Error analyzing mistakes: {e}")
            return ["Error analyzing pose. Please try again."] 