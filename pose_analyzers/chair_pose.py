"""
Chair Pose Analyzer
Converts the Flask chair pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class ChairPoseAnalyzer(BasePoseAnalyzer):
    """Chair Pose (Utkatasana) analyzer with enhanced detection"""
    
    def __init__(self):
        super().__init__()
        
        # Chair Pose specific joint indices for angle calculation
        self.joint_indices = {
            'right_elbow': 0,   # Right shoulder-elbow-wrist
            'left_elbow': 1,    # Left shoulder-elbow-wrist  
            'right_shoulder': 2, # Right elbow-shoulder-hip
            'left_shoulder': 3,  # Left elbow-shoulder-hip
            'right_hip': 4,     # Right shoulder-hip-knee
            'left_hip': 5,      # Left shoulder-hip-knee
            'right_knee': 6,    # Right hip-knee-ankle
            'left_knee': 7      # Left hip-knee-ankle
        }
        
        # Ideal angles for Chair Pose
        self.ideal_angles = {
            'right_elbow': (160, 180),    # Arms raised overhead
            'left_elbow': (160, 180),     # Arms raised overhead
            'right_shoulder': (140, 170), # Shoulder alignment
            'left_shoulder': (140, 170),  # Shoulder alignment
            'right_hip': (70, 90),        # Hip flexion for sitting
            'left_hip': (70, 90),         # Hip flexion for sitting
            'right_knee': (70, 110),      # Knee bend for chair position
            'left_knee': (70, 110)        # Knee bend for chair position
        }
    
    def get_pose_name(self) -> str:
        return "Chair Pose (Utkatasana)"
    
    def get_description(self) -> str:
        return "A powerful standing pose that strengthens the legs and core while improving balance"
    
    def get_difficulty(self) -> str:
        return "intermediate"
    
    def get_benefits(self) -> List[str]:
        return [
            'Strengthens quadriceps, glutes, and calves',
            'Improves balance and stability',
            'Engages core muscles',
            'Builds endurance and stamina',
            'Improves posture and spinal alignment'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Stand with feet hip-width apart',
            'Raise your arms overhead',
            'Bend your knees as if sitting in a chair',
            'Keep your weight in your heels',
            'Hold the position while breathing steadily',
            'Keep your torso upright and core engaged'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Chair pose and provide feedback"""
        
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
            is_chair_pose, confidence = self._classify_chair_pose(angles)
            
            final_score = int(confidence * 100)
            
            # Determine pose label
            pose_label = "Chair Pose" if is_chair_pose else "Working towards Chair Pose"
            
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
        """Calculate all relevant angles for Chair Pose analysis"""
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
    
    def _classify_chair_pose(self, angles: List[float]) -> tuple:
        """Classify if the detected pose is a proper Chair Pose"""
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
            
            overall_score = (arm_score * 0.4 + leg_score * 0.6)  # Legs more important for chair pose
            
            is_chair_pose = overall_score >= 0.7  # 70% threshold
            
            return is_chair_pose, overall_score
            
        except Exception as e:
            print(f"Error in pose classification: {e}")
            return False, 0.0
    
    def _analyze_pose_mistakes(self, angles: List[float], joint_coords: Dict[str, List[float]]) -> List[str]:
        """Analyze specific mistakes in Chair Pose and provide corrections"""
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
                        mistakes.append(f"Right Elbow: Straighten your right arm more - reach higher (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_elbow':
                        mistakes.append(f"Left Elbow: Straighten your left arm more - reach higher (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_shoulder':
                        mistakes.append(f"Right Shoulder: Lift your right arm overhead (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_shoulder':
                        mistakes.append(f"Left Shoulder: Lift your left arm overhead (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_hip':
                        mistakes.append(f"Right Hip: Sit back more - bend deeper at the hips (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_hip':
                        mistakes.append(f"Left Hip: Sit back more - bend deeper at the hips (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_knee':
                        mistakes.append(f"Right Knee: Bend your knee more - sit deeper like in a chair (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_knee':
                        mistakes.append(f"Left Knee: Bend your knee more - sit deeper like in a chair (current: {current_angle:.1f}°)")
                        
                elif current_angle > ideal_max + 10:  # Significant deviation
                    if angle_name == 'right_elbow':
                        mistakes.append(f"Right Elbow: Your right arm is too bent - extend it more (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_elbow':
                        mistakes.append(f"Left Elbow: Your left arm is too bent - extend it more (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_knee':
                        mistakes.append(f"Right Knee: Don't squat too deep - raise up slightly (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_knee':
                        mistakes.append(f"Left Knee: Don't squat too deep - raise up slightly (current: {current_angle:.1f}°)")
                    elif angle_name == 'right_shoulder':
                        mistakes.append(f"Right Shoulder: Adjust your right arm position (current: {current_angle:.1f}°)")
                    elif angle_name == 'left_shoulder':
                        mistakes.append(f"Left Shoulder: Adjust your left arm position (current: {current_angle:.1f}°)")
            
            # Additional postural checks
            if joint_coords:
                # Check if knees are going too far forward
                knee_hip_diff_r = joint_coords['right_knee'][0] - joint_coords['right_hip'][0]
                knee_hip_diff_l = joint_coords['left_knee'][0] - joint_coords['left_hip'][0]
                
                if knee_hip_diff_r > 0.1 or knee_hip_diff_l > 0.1:
                    mistakes.append("Knees: Keep your knees behind your toes - sit back more")
                
                # Check spine alignment
                shoulder_center_x = (joint_coords['left_shoulder'][0] + joint_coords['right_shoulder'][0]) / 2
                hip_center_x = (joint_coords['left_hip'][0] + joint_coords['right_hip'][0]) / 2
                
                if abs(shoulder_center_x - hip_center_x) > 0.08:
                    mistakes.append("Torso: Keep your torso upright - avoid leaning too far forward or back")
            
            if not mistakes:
                mistakes.append("Perfect Chair Pose! Excellent work!")
            
            return mistakes
            
        except Exception as e:
            print(f"Error analyzing mistakes: {e}")
            return ["Error analyzing pose. Please try again."] 