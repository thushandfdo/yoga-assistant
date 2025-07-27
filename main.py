#!/usr/bin/env python3
"""
FastAPI Yoga Assistant Backend
Unified yoga pose analysis API supporting multiple poses with authentication and session management
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
import json

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
    fullname: str
    email: str
    sex: str
    dob: str
    created_at: str



# Create FastAPI app
app = FastAPI(
    title="Yoga Assistant API",
    description="AI-powered yoga pose analysis and feedback system with session management",
    version="2.0.0",
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

# Database initialization
def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Create users table
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
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration INTEGER NOT NULL,
            avg_accuracy REAL NOT NULL,
            total_poses INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create session_poses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_poses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            pose_name TEXT NOT NULL,
            accuracy REAL NOT NULL,
            feedback TEXT,
            improvements TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

# Authentication functions
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
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id: int):
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def create_user(user: UserSignup):
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    hashed_password = get_password_hash(user.password)
    cursor.execute(
        'INSERT INTO users (fullname, email, password_hash, sex, dob) VALUES (?, ?, ?, ?, ?)',
        (user.fullname, user.email, hashed_password, user.sex, user.dob)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id = int(user_id_str)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Database startup event
@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Yoga Assistant API",
        "version": "2.0.0",
        "supported_poses": list(pose_analyzers.keys()),
        "endpoints": {
            "auth": {
                "login": "/auth/login",
                "signup": "/auth/signup",
                "me": "/auth/me"
            },
            "sessions": {
                "stats": "/api/sessions/stats",
                "list": "/api/sessions",
                "create": "/api/sessions",
                "get": "/api/sessions/{session_id}",
                "update": "/api/sessions/{session_id}",
                "delete": "/api/sessions/{session_id}"
            },
            "pose_analysis": {
                "analyze": "/api/analyze/{pose_name}",
                "analyze_enhanced": "/api/analyze-pose",
                "pose_info": "/api/poses/{pose_name}/info",
                "enhanced_pose_info": "/api/enhanced-pose-info"
            },
            "health": "/api/health",
            "poses": "/api/poses"
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

# Authentication endpoints
@app.post("/auth/signup", response_model=Token)
async def signup(user: UserSignup):
    """User signup endpoint"""
    try:
        # Check if user already exists
        existing_user = get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Create new user
        user_id = create_user(user)
        
        # Create access token
        access_token = create_access_token(data={"sub": str(user_id)})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=user_id,
            fullname=user.fullname,
            email=user.email
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """User login endpoint"""
    try:
        # Get user by email
        db_user = get_user_by_email(user.email)
        
        if not db_user or not verify_password(user.password, db_user[3]):  # password_hash is at index 3
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password"
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": str(db_user[0])})  # user_id is at index 0
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=db_user[0],
            fullname=db_user[1],
            email=db_user[2]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.get("/auth/me", response_model=UserResponse)
async def read_users_me(current_user: tuple = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        fullname=current_user[1],
        email=current_user[2],
        sex=current_user[4],
        dob=current_user[5],
        created_at=current_user[6]
    )

@app.get("/auth/protected")
async def protected_route(current_user: tuple = Depends(get_current_user)):
    """Protected route example"""
    return {"message": f"Hello {current_user[1]}, this is a protected route!"}

# Import session management functions
from session_models import (
    SessionCreate, SessionResponse, SessionsResponse, SessionCreateResponse,
    StatsResponse, DashboardStats, get_sessions_stats, get_user_sessions,
    create_session, get_session_by_id, update_session, delete_session
)

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    """Get current user ID from token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return int(user_id_str)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Session management endpoints
@app.get("/api/sessions/stats", response_model=StatsResponse)
async def get_dashboard_stats(current_user_id: int = Depends(get_current_user_id)):
    """Get dashboard statistics for the authenticated user"""
    try:
        stats = get_sessions_stats(current_user_id)
        return StatsResponse(
            success=True,
            data=DashboardStats(**stats),
            message="Statistics retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/api/sessions", response_model=SessionsResponse)
async def get_sessions(current_user_id: int = Depends(get_current_user_id)):
    """Get all sessions for the authenticated user"""
    try:
        session_responses = get_user_sessions(current_user_id)
        return SessionsResponse(
            success=True,
            data=session_responses,
            message="Sessions retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@app.post("/api/sessions", response_model=SessionCreateResponse)
async def create_session_endpoint(
    session_data: SessionCreate,
    current_user_id: int = Depends(get_current_user_id)
):
    """Create a new session for the authenticated user"""
    try:
        session_response = create_session(current_user_id, session_data)
        return SessionCreateResponse(
            success=True,
            data=session_response,
            message="Session created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/api/sessions/{session_id}", response_model=SessionCreateResponse)
async def get_session_endpoint(
    session_id: int,
    current_user_id: int = Depends(get_current_user_id)
):
    """Get a specific session by ID"""
    try:
        session_response = get_session_by_id(session_id, current_user_id)
        if not session_response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionCreateResponse(
            success=True,
            data=session_response,
            message="Session retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.put("/api/sessions/{session_id}", response_model=SessionCreateResponse)
async def update_session_endpoint(
    session_id: int,
    session_data: SessionCreate,
    current_user_id: int = Depends(get_current_user_id)
):
    """Update an existing session"""
    try:
        session_response = update_session(session_id, current_user_id, session_data)
        if not session_response:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionCreateResponse(
            success=True,
            data=session_response,
            message="Session updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(
    session_id: int,
    current_user_id: int = Depends(get_current_user_id)
):
    """Delete a session"""
    try:
        success = delete_session(session_id, current_user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"success": True, "message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")








# Existing pose analysis endpoints (preserved)
@app.get("/api/poses")
async def get_supported_poses():
    """Get list of all supported yoga poses"""
    pose_info = {}
    for pose_name, analyzer in pose_analyzers.items():
        pose_info[pose_name] = {
            "name": pose_name.replace("_", " ").title(),
            "description": f"Analysis for {pose_name.replace('_', ' ')} pose",
            "difficulty": "beginner"  # Default difficulty
        }
    return pose_info

@app.get("/api/poses/{pose_name}/info")
async def get_pose_info(pose_name: str):
    """Get detailed information about a specific pose"""
    if pose_name not in pose_analyzers:
        raise HTTPException(status_code=404, detail="Pose not found")
    
    return {
        "pose_name": pose_name,
        "display_name": pose_name.replace("_", " ").title(),
        "description": f"Detailed analysis for {pose_name.replace('_', ' ')} pose",
        "analyzer_available": True
    }

@app.post("/api/analyze/{pose_name}")
async def analyze_pose(
    pose_name: str,
    video: UploadFile = File(...),
    skill_level: Optional[str] = Form("beginner")
):
    """
    Analyze a specific yoga pose from video upload
    
    Args:
        pose_name: Name of the pose to analyze
        video: Video file upload
        skill_level: User skill level (beginner, intermediate, advanced)
    """
    if pose_name not in pose_analyzers:
        raise HTTPException(status_code=404, detail=f"Pose '{pose_name}' not supported")
    
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        
        # Extract frame from video
        frame = extract_frame_from_video(temp_video_path)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not extract frame from video")
        
        # Analyze pose
        analyzer = pose_analyzers[pose_name]
        result = analyzer.analyze_pose(frame)
        
        # Clean up temporary file
        os.unlink(temp_video_path)
        
        # Add metadata
        result["pose_name"] = pose_name
        result["skill_level"] = skill_level
        result["analysis_type"] = "specific_pose"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing pose: {str(e)}")

@app.post("/api/batch-analyze")
async def batch_analyze_poses(
    videos: List[UploadFile] = File(...),
    pose_names: List[str] = Form(...),
    skill_level: Optional[str] = Form("beginner")
):
    """
    Analyze multiple poses from multiple video uploads
    
    Args:
        videos: List of video file uploads
        pose_names: List of pose names to analyze
        skill_level: User skill level
    """
    if len(videos) != len(pose_names):
        raise HTTPException(status_code=400, detail="Number of videos must match number of pose names")
    
    results = []
    temp_files = []
    
    try:
        for i, (video, pose_name) in enumerate(zip(videos, pose_names)):
            if pose_name not in pose_analyzers:
                results.append({
                    "pose_name": pose_name,
                    "success": False,
                    "error": f"Pose '{pose_name}' not supported"
                })
                continue
            
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                content = await video.read()
                temp_video.write(content)
                temp_video_path = temp_video.name
                temp_files.append(temp_video_path)
            
            # Extract frame from video
            frame = extract_frame_from_video(temp_video_path)
            
            if frame is None:
                results.append({
                    "pose_name": pose_name,
                    "success": False,
                    "error": "Could not extract frame from video"
                })
                continue
            
            # Analyze pose
            analyzer = pose_analyzers[pose_name]
            result = analyzer.analyze_pose(frame)
            
            # Add metadata
            result["pose_name"] = pose_name
            result["skill_level"] = skill_level
            result["analysis_type"] = "batch_analysis"
            
            results.append(result)
        
        return {
            "batch_results": results,
            "total_poses": len(results),
            "successful": len([r for r in results if r.get("success", True)]),
            "failed": len([r for r in results if not r.get("success", True)])
        }
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

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

if __name__ == "__main__":
    print("=" * 60)
    print("🧘‍♀️ YOGA ASSISTANT API - FastAPI Backend v2.0")
    print("=" * 60)
    print(f"🎯 Supported Poses: {len(pose_analyzers)}")
    print(f"📊 Pose Types: {list(pose_analyzers.keys())}")
    print("🤖 Enhanced Analyzer: Hybrid ML + Rule-based Detection")
    print("🚀 Features: AI Analysis, Session Management, User Authentication")
    print("💾 Database: SQLite with session tracking")
    print("🔐 Authentication: JWT-based user management")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )