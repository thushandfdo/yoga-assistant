"""
Enhanced Pose Analyzer - Hybrid ML and Rule-based Detection
Converts yoga_1.py Flask logic to FastAPI-compatible class
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import warnings
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class EnhancedPoseAnalyzer:
    """Enhanced pose analyzer with hybrid ML and rule-based detection"""
    
    def __init__(self):
        self.confidence_threshold = 0.6
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.pose_classes = []
        self.model_trained = False
        self.model_accuracy = 0.0
        
        # Session tracking
        self.session_poses = deque(maxlen=50)
        self.detection_stats = {
            'total_detections': 0,
            'rule_based_detections': 0,
            'ml_detections': 0,
            'high_confidence_detections': 0
        }
        
        # Setup feature extraction
        self.setup_feature_extraction()
        
        # Load model
        self.load_model()
        
        print("🤖 Enhanced Hybrid Pose Analyzer initialized!")
        print(f"📊 ML Model: {'Loaded' if self.model_trained else 'Not Available'}")
        print("📏 Rule-based detection: T Pose, Side Plank, Warrior II, Plank, Mountain, Downward Dog")
    
    def setup_feature_extraction(self):
        """Setup comprehensive feature extraction"""
        self.feature_names = [
            # Basic joint angles
            'left_elbow_angle', 'right_elbow_angle',
            'left_shoulder_angle', 'right_shoulder_angle',
            'left_hip_angle', 'right_hip_angle',
            'left_knee_angle', 'right_knee_angle',
            'left_ankle_angle', 'right_ankle_angle',
            
            # Distance measurements (normalized)
            'stance_width', 'arm_span', 'knee_separation',
            'shoulder_hip_distance', 'wrist_distance',
            'shoulder_width', 'hip_width',
            
            # Body alignment features
            'torso_lean', 'hip_level_diff', 'shoulder_level_diff',
            'center_of_gravity_x', 'center_of_gravity_y',
            'body_symmetry', 'weight_balance',
            
            # Relative positions (normalized by body dimensions)
            'left_wrist_relative_x', 'left_wrist_relative_y',
            'right_wrist_relative_x', 'right_wrist_relative_y',
            'left_ankle_relative_x', 'left_ankle_relative_y',
            'right_ankle_relative_x', 'right_ankle_relative_y',
            'left_knee_relative_x', 'right_knee_relative_x',
            
            # Advanced geometric features
            'arm_symmetry', 'leg_symmetry', 'body_twist',
            'shoulder_arm_angle_left', 'shoulder_arm_angle_right',
            'hip_leg_angle_left', 'hip_leg_angle_right',
            
            # Balance and stability features
            'weight_distribution_left', 'weight_distribution_right',
            'vertical_alignment', 'horizontal_alignment',
            'pose_stability_score', 'balance_confidence'
        ]
    
    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle between three points using vector mathematics"""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return 0.0
    
    def extract_key_points(self, landmarks) -> Dict[str, List[float]]:
        """Extract key points from MediaPipe landmarks"""
        try:
            points = {
                'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
                'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
                'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
                'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
                'nose': [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            }
            return points
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return {}
    
    def extract_comprehensive_features(self, points: Dict[str, List[float]]) -> List[float]:
        """Extract comprehensive features for ML model"""
        try:
            features = []
            
            # Basic joint angles
            features.append(self.calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']))
            features.append(self.calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']))
            features.append(self.calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip']))
            features.append(self.calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip']))
            features.append(self.calculate_angle(points['left_shoulder'], points['left_hip'], points['left_knee']))
            features.append(self.calculate_angle(points['right_shoulder'], points['right_hip'], points['right_knee']))
            features.append(self.calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']))
            features.append(self.calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']))
            features.append(self.calculate_angle(points['left_knee'], points['left_ankle'], [points['left_ankle'][0], points['left_ankle'][1] - 0.1]))
            features.append(self.calculate_angle(points['right_knee'], points['right_ankle'], [points['right_ankle'][0], points['right_ankle'][1] - 0.1]))
            
            # Distance measurements (normalized by body height)
            body_height = self.calculate_distance(points['nose'], points['left_ankle'])
            if body_height > 0:
                features.append(self.calculate_distance(points['left_ankle'], points['right_ankle']) / body_height)  # stance_width
                features.append(self.calculate_distance(points['left_wrist'], points['right_wrist']) / body_height)  # arm_span
                features.append(self.calculate_distance(points['left_knee'], points['right_knee']) / body_height)  # knee_separation
                features.append(self.calculate_distance(points['left_shoulder'], points['left_hip']) / body_height)  # shoulder_hip_distance
                features.append(self.calculate_distance(points['left_wrist'], points['right_wrist']) / body_height)  # wrist_distance
                features.append(self.calculate_distance(points['left_shoulder'], points['right_shoulder']) / body_height)  # shoulder_width
                features.append(self.calculate_distance(points['left_hip'], points['right_hip']) / body_height)  # hip_width
            else:
                features.extend([0.0] * 7)
            
            # Body alignment features
            torso_center = [(points['left_shoulder'][0] + points['right_shoulder'][0]) / 2,
                           (points['left_shoulder'][1] + points['right_shoulder'][1]) / 2]
            hip_center = [(points['left_hip'][0] + points['right_hip'][0]) / 2,
                         (points['left_hip'][1] + points['right_hip'][1]) / 2]
            
            features.append(abs(torso_center[0] - hip_center[0]))  # torso_lean
            features.append(abs(points['left_hip'][1] - points['right_hip'][1]))  # hip_level_diff
            features.append(abs(points['left_shoulder'][1] - points['right_shoulder'][1]))  # shoulder_level_diff
            features.append((torso_center[0] + hip_center[0]) / 2)  # center_of_gravity_x
            features.append((torso_center[1] + hip_center[1]) / 2)  # center_of_gravity_y
            
            # Symmetry features
            left_arm_length = self.calculate_distance(points['left_shoulder'], points['left_elbow'])
            right_arm_length = self.calculate_distance(points['right_shoulder'], points['right_elbow'])
            features.append(abs(left_arm_length - right_arm_length) / max(left_arm_length, right_arm_length, 0.001))  # arm_symmetry
            
            left_leg_length = self.calculate_distance(points['left_hip'], points['left_knee'])
            right_leg_length = self.calculate_distance(points['right_hip'], points['right_knee'])
            features.append(abs(left_leg_length - right_leg_length) / max(left_leg_length, right_leg_length, 0.001))  # leg_symmetry
            
            features.append(1.0 - abs(left_arm_length - right_arm_length) / max(left_arm_length, right_arm_length, 0.001))  # body_symmetry
            features.append(0.5)  # weight_balance (placeholder)
            
            # Relative positions
            if body_height > 0:
                features.append((points['left_wrist'][0] - points['nose'][0]) / body_height)  # left_wrist_relative_x
                features.append((points['left_wrist'][1] - points['nose'][1]) / body_height)  # left_wrist_relative_y
                features.append((points['right_wrist'][0] - points['nose'][0]) / body_height)  # right_wrist_relative_x
                features.append((points['right_wrist'][1] - points['nose'][1]) / body_height)  # right_wrist_relative_y
                features.append((points['left_ankle'][0] - points['nose'][0]) / body_height)  # left_ankle_relative_x
                features.append((points['left_ankle'][1] - points['nose'][1]) / body_height)  # left_ankle_relative_y
                features.append((points['right_ankle'][0] - points['nose'][0]) / body_height)  # right_ankle_relative_x
                features.append((points['right_ankle'][1] - points['nose'][1]) / body_height)  # right_ankle_relative_y
                features.append((points['left_knee'][0] - points['nose'][0]) / body_height)  # left_knee_relative_x
                features.append((points['right_knee'][0] - points['nose'][0]) / body_height)  # right_knee_relative_x
            else:
                features.extend([0.0] * 10)
            
            # Advanced geometric features
            features.append(self.calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']))  # shoulder_arm_angle_left
            features.append(self.calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']))  # shoulder_arm_angle_right
            features.append(self.calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']))  # hip_leg_angle_left
            features.append(self.calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']))  # hip_leg_angle_right
            
            # Balance and stability features
            features.append(0.5)  # weight_distribution_left (placeholder)
            features.append(0.5)  # weight_distribution_right (placeholder)
            features.append(1.0 - abs(torso_center[0] - hip_center[0]))  # vertical_alignment
            features.append(1.0 - abs(points['left_shoulder'][1] - points['right_shoulder'][1]))  # horizontal_alignment
            features.append(0.8)  # pose_stability_score (placeholder)
            features.append(0.7)  # balance_confidence (placeholder)
            
            return features
            
        except Exception as e:
            print(f"Error extracting comprehensive features: {e}")
            return [0.0] * len(self.feature_names)
    
    def detect_rule_based_poses(self, points: Dict[str, List[float]]) -> Tuple[str, float, str]:
        """Detect poses using rule-based logic"""
        try:
            # Calculate key angles
            left_elbow_angle = self.calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist'])
            right_elbow_angle = self.calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist'])
            left_shoulder_angle = self.calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip'])
            right_shoulder_angle = self.calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip'])
            left_hip_angle = self.calculate_angle(points['left_shoulder'], points['left_hip'], points['left_knee'])
            right_hip_angle = self.calculate_angle(points['right_shoulder'], points['right_hip'], points['right_knee'])
            left_knee_angle = self.calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle'])
            right_knee_angle = self.calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle'])
            
            # Calculate distances
            stance_width = self.calculate_distance(points['left_ankle'], points['right_ankle'])
            arm_span = self.calculate_distance(points['left_wrist'], points['right_wrist'])
            shoulder_width = self.calculate_distance(points['left_shoulder'], points['right_shoulder'])
            
            # T Pose detection
            if (150 <= left_elbow_angle <= 180 and 150 <= right_elbow_angle <= 180 and
                150 <= left_shoulder_angle <= 180 and 150 <= right_shoulder_angle <= 180 and
                160 <= left_hip_angle <= 180 and 160 <= right_hip_angle <= 180 and
                160 <= left_knee_angle <= 180 and 160 <= right_knee_angle <= 180 and
                arm_span > shoulder_width * 1.5):
                return "T Pose", 0.92, "rule_based"
            
            # Side Plank detection (Left)
            if (80 <= left_elbow_angle <= 120 and 80 <= right_elbow_angle <= 120 and
                abs(points['left_shoulder'][1] - points['right_shoulder'][1]) < 0.1 and
                abs(points['left_hip'][1] - points['right_hip'][1]) < 0.1):
                return "Left Side Plank", 0.88, "rule_based"
            
            # Side Plank detection (Right)
            if (80 <= left_elbow_angle <= 120 and 80 <= right_elbow_angle <= 120 and
                abs(points['left_shoulder'][1] - points['right_shoulder'][1]) < 0.1 and
                abs(points['left_hip'][1] - points['right_hip'][1]) < 0.1):
                return "Right Side Plank", 0.88, "rule_based"
            
            # Warrior II detection
            if (stance_width > shoulder_width * 1.8 and
                80 <= left_knee_angle <= 120 and 160 <= right_knee_angle <= 180 and
                150 <= left_elbow_angle <= 180 and 150 <= right_elbow_angle <= 180):
                return "Warrior II Pose", 0.90, "rule_based"
            
            # Plank detection
            if (80 <= left_elbow_angle <= 120 and 80 <= right_elbow_angle <= 120 and
                abs(points['left_shoulder'][1] - points['right_shoulder'][1]) < 0.05 and
                abs(points['left_hip'][1] - points['right_hip'][1]) < 0.05):
                return "Plank Pose", 0.87, "rule_based"
            
            # Mountain Pose detection
            if (160 <= left_elbow_angle <= 180 and 160 <= right_elbow_angle <= 180 and
                160 <= left_hip_angle <= 180 and 160 <= right_hip_angle <= 180 and
                160 <= left_knee_angle <= 180 and 160 <= right_knee_angle <= 180 and
                stance_width < shoulder_width * 1.2):
                return "Mountain Pose", 0.85, "rule_based"
            
            # Downward Dog detection
            if (80 <= left_elbow_angle <= 120 and 80 <= right_elbow_angle <= 120 and
                80 <= left_hip_angle <= 120 and 80 <= right_hip_angle <= 120 and
                points['left_hip'][1] > points['left_shoulder'][1] and
                points['right_hip'][1] > points['right_shoulder'][1]):
                return "Downward Dog", 0.86, "rule_based"
            
            return "Unknown Pose", 0.0, "rule_based"
            
        except Exception as e:
            print(f"Error in rule-based detection: {e}")
            return "Unknown Pose", 0.0, "rule_based"
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'enhanced_yoga_model_images.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.ml_model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.pose_classes = model_data['pose_classes']
                self.model_accuracy = model_data.get('model_accuracy', 0.0)
                
                print("📥 ML Model loaded successfully!")
                print(f"   Pose classes: {self.pose_classes}")
                print(f"   Model accuracy: {self.model_accuracy:.1%}")
                
                self.model_trained = True
                return True
            else:
                print("📝 No pre-trained ML model found. Using rule-based detection only.")
                self.model_trained = False
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_trained = False
            return False
    
    def predict_pose_ml(self, points: Dict[str, List[float]]) -> Tuple[str, float, str]:
        """Predict pose using ML model"""
        if not self.model_trained or not self.ml_model:
            return None, 0.0, "ml"
        
        try:
            features = self.extract_comprehensive_features(points)
            features_scaled = self.scaler.transform([features])
            
            # Get prediction and confidence
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence, "ml"
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return None, 0.0, "ml"
    
    def classify_pose_hybrid(self, points: Dict[str, List[float]]) -> Tuple[str, float, str, Dict]:
        """Hybrid classification using both ML and rule-based methods"""
        if not points:
            return "Unknown Pose", 0.0, "none", {}
        
        # First try rule-based detection for specific poses
        rule_pose, rule_confidence, rule_method = self.detect_rule_based_poses(points)
        
        # Try ML prediction
        ml_pose, ml_confidence, ml_method = None, 0.0, "ml"
        if self.model_trained:
            ml_pose, ml_confidence, ml_method = self.predict_pose_ml(points)
        
        # Determine best result and create detailed response
        detection_details = {
            'rule_based': {
                'pose': rule_pose,
                'confidence': rule_confidence,
                'available': True
            },
            'ml_based': {
                'pose': ml_pose,
                'confidence': ml_confidence,
                'available': self.model_trained
            }
        }
        
        # Decision logic for hybrid classification
        if rule_confidence > 0.85:
            # High confidence rule-based detection
            final_pose = rule_pose
            final_confidence = rule_confidence
            final_method = "rule_based"
        elif ml_confidence > rule_confidence and ml_confidence > 0.7:
            # High confidence ML detection
            final_pose = ml_pose
            final_confidence = ml_confidence
            final_method = "ml"
        elif rule_confidence > 0.5:
            # Medium confidence rule-based
            final_pose = rule_pose
            final_confidence = rule_confidence
            final_method = "rule_based"
        elif ml_confidence > 0.5:
            # Medium confidence ML
            final_pose = ml_pose
            final_confidence = ml_confidence * 0.95  # Slight discount
            final_method = "ml"
        elif rule_confidence > 0.3:
            # Low confidence rule-based
            final_pose = rule_pose
            final_confidence = rule_confidence
            final_method = "rule_based"
        else:
            # No confident detection
            final_pose = "Unknown Pose"
            final_confidence = 0.0
            final_method = "none"
        
        return final_pose, final_confidence, final_method, detection_details
    
    def analyze_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze pose using hybrid detection system"""
        try:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process the image
                results = pose.process(image_rgb)
                
                if not results.pose_landmarks:
                    return {
                        'success': False,
                        'message': 'No pose detected. Please ensure you are visible in the frame.',
                        'detection_method': 'none',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Extract key points
                points = self.extract_key_points(results.pose_landmarks.landmark)
                
                if not points:
                    return {
                        'success': False,
                        'message': 'Could not extract pose landmarks properly.',
                        'detection_method': 'none',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Classify pose using hybrid system
                pose_name, confidence, method, detection_details = self.classify_pose_hybrid(points)
                
                # Update session tracking
                self.session_poses.append({
                    'pose': pose_name,
                    'confidence': confidence,
                    'method': method,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update detection stats
                self.detection_stats['total_detections'] += 1
                if method == 'rule_based':
                    self.detection_stats['rule_based_detections'] += 1
                elif method == 'ml':
                    self.detection_stats['ml_detections'] += 1
                if confidence > 0.8:
                    self.detection_stats['high_confidence_detections'] += 1
                
                # Create response
                response = {
                    'success': True,
                    'pose_detected': pose_name,
                    'confidence': confidence,
                    'detection_method': method,
                    'detection_details': detection_details,
                    'session_stats': self.detection_stats,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add feedback based on pose
                if pose_name != "Unknown Pose":
                    response['feedback'] = [f"Great! {pose_name} detected with {confidence:.1%} confidence."]
                    response['score'] = int(confidence * 100)
                else:
                    response['feedback'] = ["No specific pose detected. Try adjusting your position."]
                    response['score'] = 0
                
                return response
                
        except Exception as e:
            print(f"Error in pose analysis: {e}")
            return {
                'success': False,
                'message': f'Analysis error: {str(e)}',
                'detection_method': 'none',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_supported_poses(self) -> Dict[str, Any]:
        """Get list of supported poses"""
        rule_based_poses = [
            'T Pose', 'Left Side Plank', 'Right Side Plank', 
            'Warrior II Pose', 'Plank Pose', 'Mountain Pose', 'Downward Dog'
        ]
        
        ml_poses = self.pose_classes if self.model_trained else []
        
        return {
            'rule_based_poses': rule_based_poses,
            'ml_poses': ml_poses,
            'total_poses': len(rule_based_poses) + len(ml_poses),
            'hybrid_detection': True
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary and statistics"""
        if not self.session_poses:
            return {'message': 'No poses detected in this session.'}
        
        pose_counts = {}
        method_counts = {'rule_based': 0, 'ml': 0, 'none': 0}
        total_confidence = 0
        
        for pose_data in self.session_poses:
            pose = pose_data['pose']
            method = pose_data['method']
            confidence = pose_data['confidence']
            
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(self.session_poses) if self.session_poses else 0
        
        return {
            'total_detections': len(self.session_poses),
            'pose_distribution': pose_counts,
            'method_distribution': method_counts,
            'average_confidence': avg_confidence,
            'detection_stats': self.detection_stats
        } 