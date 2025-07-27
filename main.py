#!/usr/bin/env python3
"""
FastAPI Yoga Assistant Backend
Unified yoga pose analysis API supporting multiple poses with authentication
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import tempfile
import os
from typing import List, Optional
import time
import sqlite3
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
import jwt

# Import pose analyzers
from pose_analyzers.chair_pose import ChairPoseAnalyzer
from pose_analyzers.mountain_pose import MountainPoseAnalyzer
from pose_analyzers.tree_pose import TreePoseAnalyzer
from pose_analyzers.downdog_pose import DowndogPoseAnalyzer
from pose_analyzers.goddess_pose import GoddessPoseAnalyzer
from pose_analyzers.lord_pose import LordPoseAnalyzer
from pose_analyzers.lowlung_pose import LowlungPoseAnalyzer
from pose_analyzers.side_plank import SidePlankAnalyzer
from pose_analyzers.staff_pose import StaffPoseAnalyzer
from pose_analyzers.t_pose import TPoseAnalyzer
from pose_analyzers.warrior2_pose import Warrior2PoseAnalyzer
from pose_analyzers.enhanced_pose_analyzer import EnhancedPoseAnalyzer
from utils.video_processor import extract_frame_from_video

# Authentication Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Enums
class Sex(str, Enum):
    male = "male"
    female = "female"
    other = "other"

# Authentication Pydantic models
class UserSignup(BaseModel):
    fullname: str
    email: EmailStr
    password: str
    sex: Sex
    dob: str  # Format: YYYY-MM-DD
    
    @validator('fullname')
    def validate_fullname(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Full name must be at least 2 characters long')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v
    
    @validator('dob')
    def validate_dob(cls, v):
        try:
            dob_date = datetime.strptime(v, '%Y-%m-%d').date()
            today = datetime.now().date()
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
            if age < 13:
                raise ValueError('User must be at least 13 years old')
            if age > 120:
                raise ValueError('Invalid date of birth')
        except ValueError as e:
            if 'time data' in str(e):
                raise ValueError('Date must be in YYYY-MM-DD format')
            raise e
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    fullname: str
    email: str

class UserResponse(BaseModel):
    id: int
    fullname: str
    email: str
    sex: str
    dob: str
    created_at: str

# Create FastAPI app
app = FastAPI(
    title="Yoga Assistant API",
    description="AI-powered yoga pose analysis and feedback system with authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            sex TEXT NOT NULL,
            dob TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Authentication utility functions
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_email(email: str):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id: int):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(user: UserSignup):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (fullname, email, password_hash, sex, dob)
            VALUES (?, ?, ?, ?, ?)
        ''', (user.fullname, user.email, get_password_hash(user.password), user.sex, user.dob))
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    finally:
        conn.close()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Initialize pose analyzers
pose_analyzers = {
    "chair": ChairPoseAnalyzer(),
    "mountain": MountainPoseAnalyzer(),
    "tree": TreePoseAnalyzer(),
    "downdog": DowndogPoseAnalyzer(),
    "goddess": GoddessPoseAnalyzer(),
    "lord": LordPoseAnalyzer(),
    "lowlung": LowlungPoseAnalyzer(),
    "side_plank": SidePlankAnalyzer(),
    "staff": StaffPoseAnalyzer(),
    "t_pose": TPoseAnalyzer(),
    "warrior2": Warrior2PoseAnalyzer(),
}

# Initialize enhanced pose analyzer for hybrid detection
enhanced_analyzer = EnhancedPoseAnalyzer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Yoga Assistant API",
        "version": "1.0.0",
        "supported_poses": list(pose_analyzers.keys()),
        "endpoints": {
            "analyze": "/api/analyze/{pose_name}",
            "analyze_enhanced": "/api/analyze-pose",
            "pose_info": "/api/poses/{pose_name}/info",
            "enhanced_pose_info": "/api/enhanced-pose-info",
            "health": "/api/health",
            "poses": "/api/poses",
            "auth": {
                "signup": "/auth/signup",
                "login": "/auth/login",
                "me": "/auth/me",
                "protected": "/auth/protected"
            }
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Yoga Assistant API is running",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "available_poses": len(pose_analyzers),
        "poses": list(pose_analyzers.keys())
    }

@app.get("/api/poses")
async def get_supported_poses():
    """Get list of all supported yoga poses"""
    pose_info = {}
    for pose_name, analyzer in pose_analyzers.items():
        pose_info[pose_name] = {
            "name": analyzer.get_pose_name(),
            "description": analyzer.get_description(),
            "difficulty": analyzer.get_difficulty(),
            "endpoint": f"/api/analyze/{pose_name}"
        }
    
    return {
        "supported_poses": pose_info,
        "total_count": len(pose_analyzers)
    }

@app.get("/api/poses/{pose_name}/info")
async def get_pose_info(pose_name: str):
    """Get detailed information about a specific pose"""
    if pose_name not in pose_analyzers:
        raise HTTPException(status_code=404, detail=f"Pose '{pose_name}' not supported")
    
    analyzer = pose_analyzers[pose_name]
    return analyzer.get_pose_info()

@app.post("/api/analyze/{pose_name}")
async def analyze_pose(
    pose_name: str,
    video: UploadFile = File(...),
    skill_level: Optional[str] = Form("beginner")
):
    """
    Analyze uploaded video for specific yoga pose
    
    Args:
        pose_name: Name of the pose to analyze (chair, mountain, tree, etc.)
        video: Video file to analyze
        skill_level: User's skill level (beginner, intermediate, advanced)
    """
    
    if pose_name not in pose_analyzers:
        raise HTTPException(
            status_code=404, 
            detail=f"Pose '{pose_name}' not supported. Supported poses: {list(pose_analyzers.keys())}"
        )
    
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # Validate file type
    allowed_extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv'}
    file_ext = os.path.splitext(video.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    # Save video to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            contents = await video.read()
            temp_file.write(contents)
            temp_path = temp_file.name
        
        try:
            # Extract frame from video
            frame = extract_frame_from_video(temp_path)
            
            if frame is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not extract frame from video. Please ensure video is valid."
                )
            
            # Analyze the pose
            analyzer = pose_analyzers[pose_name]
            result = analyzer.analyze_pose(frame, skill_level)
            
            # Add metadata
            result.update({
                "pose_name": pose_name,
                "skill_level": skill_level,
                "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "video_filename": video.filename
            })
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/api/batch-analyze")
async def batch_analyze_poses(
    videos: List[UploadFile] = File(...),
    pose_names: List[str] = Form(...),
    skill_level: Optional[str] = Form("beginner")
):
    """
    Batch analyze multiple videos for different poses
    
    Args:
        videos: List of video files to analyze
        pose_names: List of pose names corresponding to each video
        skill_level: User's skill level
    """
    
    if len(videos) != len(pose_names):
        raise HTTPException(
            status_code=400, 
            detail="Number of videos must match number of pose names"
        )
    
    if len(videos) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 videos allowed per batch"
        )
    
    results = []
    
    for i, (video, pose_name) in enumerate(zip(videos, pose_names)):
        try:
            # Reuse the single analysis endpoint logic
            result = await analyze_pose(pose_name, video, skill_level)
            result["batch_index"] = i
            results.append(result)
            
        except HTTPException as e:
            results.append({
                "batch_index": i,
                "pose_name": pose_name,
                "video_filename": video.filename,
                "error": e.detail,
                "success": False
            })
        except Exception as e:
            results.append({
                "batch_index": i,
                "pose_name": pose_name,
                "video_filename": video.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "batch_results": results,
        "total_analyzed": len(results),
        "successful": len([r for r in results if r.get("success", True)]),
        "failed": len([r for r in results if not r.get("success", True)])
    }

from pydantic import BaseModel

class AnalyzePoseRequest(BaseModel):
    image: str
    skill_level: Optional[str] = "beginner"

@app.post("/api/analyze-pose")
async def analyze_pose_enhanced(
    request: AnalyzePoseRequest
):
    """
    Enhanced pose analysis with hybrid ML and rule-based detection
    
    This endpoint uses the enhanced pose analyzer that combines:
    - Rule-based detection for specific poses (T Pose, Side Plank, Warrior II, etc.)
    - ML model detection for trained poses
    - Hybrid decision logic for optimal accuracy
    
    Args:
        request: JSON object containing image (base64) and skill_level
    """
    try:
        import base64
        import cv2
        import numpy as np
        
        # Parse base64 image data
        if request.image.startswith('data:image'):
            # Remove data URL prefix
            image_data = request.image.split(',')[1]
        else:
            image_data = request.image
            
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please ensure the image is valid."
            )
        
        # Analyze pose using enhanced analyzer
        result = enhanced_analyzer.analyze_pose(frame)
        
        # Add additional metadata
        result["skill_level"] = request.skill_level
        result["analysis_type"] = "enhanced_hybrid"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing pose: {str(e)}"
        )

@app.get("/api/enhanced-pose-info")
async def get_enhanced_pose_info():
    """Get information about the enhanced pose analyzer"""
    supported_poses = enhanced_analyzer.get_supported_poses()
    session_summary = enhanced_analyzer.get_session_summary()
    
    return {
        "analyzer_type": "Enhanced Hybrid Pose Analyzer",
        "description": "Combines rule-based and ML detection for maximum accuracy",
        "supported_poses": supported_poses,
        "session_summary": session_summary,
        "detection_methods": {
            "rule_based": "Mathematical angle and distance analysis",
            "ml_based": "Trained RandomForest model with comprehensive features",
            "hybrid": "Intelligent combination of both methods"
        },
        "features": [
            "Real-time pose landmark extraction",
            "Comprehensive angle calculations", 
            "Body alignment analysis",
            "Confidence-based pose selection",
            "Session tracking and statistics",
            "Multi-method pose validation"
        ]
    }

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get summary analytics for the API"""
    return {
        "total_poses_supported": len(pose_analyzers),
        "pose_categories": {
            "standing": ["mountain", "tree", "warrior2"],
            "sitting": ["staff", "chair"],
            "balancing": ["tree", "side_plank"],
            "strength": ["chair", "side_plank", "downdog"],
            "flexibility": ["goddess", "lord", "lowlung"]
        },
        "difficulty_levels": ["beginner", "intermediate", "advanced"],
        "analysis_features": [
            "Real-time pose detection",
            "Angle measurement",
            "Balance assessment", 
            "Personalized feedback",
            "Skill-level adaptation"
        ]
    }

# Authentication Endpoints
@app.post("/auth/signup", response_model=Token)
async def signup(user: UserSignup):
    """
    Register a new user
    
    Args:
        user: User registration data including fullname, email, password, sex, and dob
    """
    # Check if user already exists
    existing_user = get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_id = create_user(user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "fullname": user.fullname,
        "email": user.email
    }

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """
    Authenticate user and return access token
    
    Args:
        user: User login credentials (email and password)
    """
    # Authenticate user
    db_user = get_user_by_email(user.email)
    if not db_user or not verify_password(user.password, db_user[3]):  # password_hash is at index 3
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(db_user[0])}, expires_delta=access_token_expires  # user id is at index 0
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": db_user[0],
        "fullname": db_user[1],
        "email": db_user[2]
    }

@app.get("/auth/me", response_model=UserResponse)
async def read_users_me(current_user: tuple = Depends(get_current_user)):
    """
    Get current user information
    
    Args:
        current_user: Current authenticated user (injected via dependency)
    """
    return {
        "id": current_user[0],
        "fullname": current_user[1],
        "email": current_user[2],
        "sex": current_user[4],
        "dob": current_user[5],
        "created_at": current_user[6]
    }

@app.get("/auth/protected")
async def protected_route(current_user: tuple = Depends(get_current_user)):
    """
    Example protected route that requires authentication
    
    Args:
        current_user: Current authenticated user (injected via dependency)
    """
    return {
        "message": f"Hello {current_user[1]}, this is a protected route!",
        "user_id": current_user[0]
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🧘‍♀️ YOGA ASSISTANT API - FastAPI Backend")
    print("=" * 60)
    print(f"🎯 Supported Poses: {len(pose_analyzers)}")
    print(f"📊 Pose Types: {list(pose_analyzers.keys())}")
    print("🤖 Enhanced Analyzer: Hybrid ML + Rule-based Detection")
    print("🚀 Features: AI Analysis, Real-time Feedback, Multi-pose Support")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )