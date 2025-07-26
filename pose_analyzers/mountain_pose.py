"""
Mountain Pose Analyzer
Converts the Flask mountain pose analysis to FastAPI format
"""

import numpy as np
from typing import Dict, List, Any
from .base_analyzer import BasePoseAnalyzer


class MountainPoseAnalyzer(BasePoseAnalyzer):
    """Mountain Pose (Tadasana) analyzer with enhanced detection"""
    
    def __init__(self):
        super().__init__()
    
    def get_pose_name(self) -> str:
        return "Mountain Pose (Tadasana)"
    
    def get_description(self) -> str:
        return "Stand tall with feet together, arms at sides, spine straight"
    
    def get_difficulty(self) -> str:
        return "beginner"
    
    def get_benefits(self) -> List[str]:
        return [
            'Improves posture and body awareness',
            'Strengthens core and leg muscles',
            'Develops balance and stability',
            'Calms the mind and promotes focus',
            'Foundation for all standing poses'
        ]
    
    def get_instructions(self) -> List[str]:
        return [
            'Stand with feet together or hip-width apart',
            'Let arms hang naturally at your sides',
            'Keep shoulders relaxed and down',
            'Lengthen your spine upward',
            'Engage your core muscles gently',
            'Breathe deeply and feel grounded'
        ]
    
    def get_common_mistakes(self) -> List[str]:
        return [
            'Tensing shoulders or arms',
            'Leaning to one side',
            'Locking knees too hard',
            'Holding breath',
            'Arching back excessively'
        ]
    
    def analyze_pose(self, image: np.ndarray, skill_level: str = "beginner") -> Dict[str, Any]:
        """Analyze Mountain Pose and provide feedback"""
        
        # Detect pose landmarks
        results = self.detect_pose_landmarks(image)
        
        if not results or not results.pose_landmarks:
            return self.create_error_response(
                'No pose detected. Please ensure you are visible in the frame.'
            )
        
        landmarks = results.pose_landmarks.landmark
        
        try:
            # Extract key points
            points = self.extract_landmarks(landmarks)
            if not points:
                return self.create_error_response(
                    'Could not extract pose keypoints.'
                )
            
            # Calculate features
            features = self._calculate_pose_features(points)
            if not features:
                return self.create_error_response(
                    'Could not calculate pose features.'
                )
            
            # Get baseline for comparison
            baseline_pose = self._get_mountain_pose_baseline()
            
            # Analyze and provide feedback
            feedback = []
            scores = []
            
            # Define angle tolerances for Mountain Pose
            angle_tolerances = {
                'right_elbow_angle': 8,
                'left_elbow_angle': 8,
                'right_shoulder_angle': 15,
                'left_shoulder_angle': 15,
                'right_hip_angle': 8,
                'left_hip_angle': 8,
                'right_knee_angle': 5,
                'left_knee_angle': 5,
                'spine_vertical': 5,
                'shoulder_level': 0.03,
                'hip_level': 0.02,
                'weight_distribution': 0.05,
                'posture_alignment': 0.08
            }
            
            # Analysis features
            key_features = [
                ('right_elbow_angle', 'Right Arm Position'),
                ('left_elbow_angle', 'Left Arm Position'),
                ('right_shoulder_angle', 'Right Shoulder Relaxation'),
                ('left_shoulder_angle', 'Left Shoulder Relaxation'),
                ('right_hip_angle', 'Right Hip Alignment'),
                ('left_hip_angle', 'Left Hip Alignment'),
                ('right_knee_angle', 'Right Leg Straightness'),
                ('left_knee_angle', 'Left Leg Straightness'),
                ('spine_vertical', 'Spine Alignment'),
                ('shoulder_level', 'Shoulder Level'),
                ('hip_level', 'Hip Level'),
                ('weight_distribution', 'Weight Balance'),
                ('posture_alignment', 'Overall Posture')
            ]
            
            for feature_key, description in key_features:
                if feature_key in baseline_pose and feature_key in features:
                    baseline_stats = baseline_pose[feature_key]
                    user_value = features[feature_key]
                    baseline_mean = baseline_stats['mean']
                    baseline_std = baseline_stats['std']
                    
                    # Calculate z-score for scoring
                    if baseline_std > 0:
                        z_score = abs(user_value - baseline_mean) / baseline_std
                        feature_score = max(0, 100 - (z_score * 15))
                        scores.append(feature_score)
                        
                        # Generate specific Mountain Pose feedback
                        tolerance = angle_tolerances.get(feature_key, 10)
                        
                        # Arms hanging naturally feedback
                        if feature_key == 'right_shoulder_angle':
                            if user_value > baseline_mean + tolerance:
                                feedback.append("Right Shoulder: Relax your right arm - let it hang naturally at your side")
                        elif feature_key == 'left_shoulder_angle':
                            if user_value > baseline_mean + tolerance:
                                feedback.append("Left Shoulder: Relax your left arm - let it hang naturally at your side")
                        
                        # Elbow position feedback  
                        elif feature_key == 'right_elbow_angle':
                            if user_value < baseline_mean - tolerance:
                                feedback.append("Right Arm: Straighten your right arm more - avoid bending at elbow")
                        elif feature_key == 'left_elbow_angle':
                            if user_value < baseline_mean - tolerance:
                                feedback.append("Left Arm: Straighten your left arm more - avoid bending at elbow")
                        
                        # Hip alignment feedback
                        elif feature_key in ['right_hip_angle', 'left_hip_angle']:
                            if user_value < baseline_mean - tolerance:
                                feedback.append("Hip Alignment: Stand taller - lengthen your spine")
                        
                        # Leg straightness feedback
                        elif feature_key in ['right_knee_angle', 'left_knee_angle']:
                            side = 'right' if 'right' in feature_key else 'left'
                            if user_value < baseline_mean - tolerance:
                                feedback.append(f"Leg Position: Straighten your {side} leg more - engage thigh muscles")
                        
                        # Spine alignment feedback (very important for Mountain Pose)
                        elif feature_key == 'spine_vertical':
                            if abs(user_value - baseline_mean) > tolerance:
                                feedback.append("Spine Alignment: Stand taller - imagine a string pulling you up from the crown of your head")
                        
                        # Balance and symmetry feedback
                        elif feature_key == 'shoulder_level':
                            if user_value > baseline_mean + tolerance:
                                feedback.append("Shoulder Level: Level your shoulders - relax and even out both sides")
                        elif feature_key == 'weight_distribution':
                            if user_value > baseline_mean + tolerance:
                                feedback.append("Balance: Distribute weight evenly between both feet")
                        elif feature_key == 'posture_alignment':
                            if user_value > baseline_mean + tolerance:
                                feedback.append("Posture: Center your head over your body - avoid leaning")
            
            # Calculate overall score
            final_score = int(np.mean(scores)) if scores else 0
            
            # Classify pose
            pose_label = "Mountain Pose"
            if (features['spine_vertical'] >= 85 and features['spine_vertical'] <= 95 and
                features['right_elbow_angle'] >= 170 and features['left_elbow_angle'] >= 170 and
                features['right_shoulder_angle'] <= 25 and features['left_shoulder_angle'] <= 25):
                pose_label = "Mountain Pose"
            else:
                pose_label = "Working towards Mountain Pose"
            
            # Add overall encouragement
            if final_score >= 95:
                feedback.insert(0, "Perfect Mountain Pose! You embody stillness and strength!")
            elif final_score >= 85:
                feedback.insert(0, "Excellent Mountain Pose - strong and steady!")
            elif final_score >= 75:
                feedback.insert(0, "Great Mountain Pose foundation, minor adjustments needed")
            elif final_score >= 65:
                feedback.insert(0, "Good progress - focus on spine alignment")
            else:
                feedback.insert(0, "Practice standing tall with arms at sides")
            
            # Create angle dictionary
            angle_dict = {
                'right_elbow': int(features['right_elbow_angle']),
                'left_elbow': int(features['left_elbow_angle']),
                'right_shoulder': int(features['right_shoulder_angle']),
                'left_shoulder': int(features['left_shoulder_angle']),
                'right_hip': int(features['right_hip_angle']),
                'left_hip': int(features['left_hip_angle']),
                'right_knee': int(features['right_knee_angle']),
                'left_knee': int(features['left_knee_angle']),
                'spine_vertical': int(features['spine_vertical'])
            }
            
            # Create additional metrics
            additional_metrics = {
                'shoulder_level': round(features['shoulder_level'], 3),
                'hip_level': round(features['hip_level'], 3),
                'weight_distribution': round(features['weight_distribution'], 3),
                'posture_alignment': round(features['posture_alignment'], 3)
            }
            
            return self.create_success_response(
                pose_detected=pose_label,
                score=final_score,
                feedback=feedback if feedback else ["Perfect Mountain Pose! Well done!"],
                angles=angle_dict,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            return self.create_error_response(
                f'Error analyzing pose: {str(e)}'
            )
    
    def _calculate_pose_features(self, points: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate comprehensive Mountain Pose features"""
        if not points:
            return {}
            
        features = {}
        
        try:
            # === CORE MOUNTAIN POSE ANGLES ===
            # Arm angles (arms hanging naturally at sides)
            features['right_elbow_angle'] = self.calculate_angle(
                points['right_shoulder'], points['right_elbow'], points['right_wrist'])
            features['left_elbow_angle'] = self.calculate_angle(
                points['left_shoulder'], points['left_elbow'], points['left_wrist'])
            
            # Shoulder angles (arms at sides)
            features['right_shoulder_angle'] = self.calculate_angle(
                points['right_elbow'], points['right_shoulder'], points['right_hip'])
            features['left_shoulder_angle'] = self.calculate_angle(
                points['left_elbow'], points['left_shoulder'], points['left_hip'])
            
            # Hip alignment (straight upright posture)
            features['right_hip_angle'] = self.calculate_angle(
                points['right_shoulder'], points['right_hip'], points['right_knee'])
            features['left_hip_angle'] = self.calculate_angle(
                points['left_shoulder'], points['left_hip'], points['left_knee'])
            
            # Leg straightness
            features['right_knee_angle'] = self.calculate_angle(
                points['right_hip'], points['right_knee'], points['right_ankle'])
            features['left_knee_angle'] = self.calculate_angle(
                points['left_hip'], points['left_knee'], points['left_ankle'])
            
            # === MOUNTAIN POSE SPECIFIC ALIGNMENT ===
            # Spine alignment (very important for Mountain Pose)
            features['spine_vertical'] = self.calculate_angle(
                points['nose'], 
                [(points['left_shoulder'][0] + points['right_shoulder'][0])/2,
                 (points['left_shoulder'][1] + points['right_shoulder'][1])/2], 
                [(points['left_hip'][0] + points['right_hip'][0])/2,
                 (points['left_hip'][1] + points['right_hip'][1])/2])
            
            # Feet alignment (parallel feet)
            features['feet_parallel'] = abs(points['left_ankle'][1] - points['right_ankle'][1]) * 180
            
            # === SYMMETRY AND BALANCE ===
            # Shoulder levelness (critical for Mountain Pose)
            features['shoulder_level'] = abs(points['left_shoulder'][1] - points['right_shoulder'][1])
            
            # Hip levelness
            features['hip_level'] = abs(points['left_hip'][1] - points['right_hip'][1])
            
            # === WEIGHT DISTRIBUTION ===
            # Center of gravity
            center_x = (points['left_hip'][0] + points['right_hip'][0]) / 2
            center_y = (points['left_hip'][1] + points['right_hip'][1]) / 2
            
            # Weight distribution between feet
            ankle_center_x = (points['left_ankle'][0] + points['right_ankle'][0]) / 2
            features['weight_distribution'] = abs(center_x - ankle_center_x)
            
            # === OVERALL POSTURE ALIGNMENT ===
            # Head alignment over body center
            features['posture_alignment'] = self.calculate_distance(
                points['nose'], [center_x, points['nose'][1]])
            
            # Arm symmetry
            features['arm_symmetry'] = abs(features['right_elbow_angle'] - features['left_elbow_angle'])
            
            # Stance width (feet should be hip-width apart)
            features['stance_width'] = self.calculate_distance(points['left_ankle'], points['right_ankle'])
            
            return features
            
        except Exception as e:
            print(f"Error calculating pose features: {e}")
            return {}
    
    def _get_mountain_pose_baseline(self) -> Dict[str, Dict[str, float]]:
        """Get ideal Mountain Pose baseline statistics"""
        return {
            'right_elbow_angle': {'mean': 178.0, 'std': 6.2, 'min': 165, 'max': 185},
            'left_elbow_angle': {'mean': 178.5, 'std': 6.0, 'min': 167, 'max': 185},
            'right_shoulder_angle': {'mean': 12.5, 'std': 4.8, 'min': 5, 'max': 22},
            'left_shoulder_angle': {'mean': 12.8, 'std': 4.6, 'min': 6, 'max': 21},
            'right_hip_angle': {'mean': 180.2, 'std': 3.2, 'min': 175, 'max': 185},
            'left_hip_angle': {'mean': 180.0, 'std': 3.1, 'min': 176, 'max': 184},
            'right_knee_angle': {'mean': 179.8, 'std': 2.8, 'min': 175, 'max': 185},
            'left_knee_angle': {'mean': 179.6, 'std': 2.9, 'min': 174, 'max': 185},
            'spine_vertical': {'mean': 90.0, 'std': 2.5, 'min': 86, 'max': 94},
            'shoulder_level': {'mean': 0.015, 'std': 0.012, 'min': 0.0, 'max': 0.04},
            'hip_level': {'mean': 0.010, 'std': 0.008, 'min': 0.0, 'max': 0.03},
            'weight_distribution': {'mean': 0.025, 'std': 0.018, 'min': 0.0, 'max': 0.06},
            'posture_alignment': {'mean': 0.035, 'std': 0.022, 'min': 0.0, 'max': 0.08}
        } 