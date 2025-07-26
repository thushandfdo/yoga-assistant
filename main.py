#!/usr/bin/env python3
"""
FastAPI Yoga Assistant Backend
Unified yoga pose analysis API supporting multiple poses
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from typing import List, Optional
import time

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
from utils.video_processor import extract_frame_from_video

# Create FastAPI app
app = FastAPI(
    title="Yoga Assistant API",
    description="AI-powered yoga pose analysis and feedback system",
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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Yoga Assistant API",
        "version": "1.0.0",
        "supported_poses": list(pose_analyzers.keys()),
        "endpoints": {
            "analyze": "/api/analyze/{pose_name}",
            "pose_info": "/api/poses/{pose_name}/info",
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
    print("🧘‍♀️ YOGA ASSISTANT API - FastAPI Backend")
    print("=" * 60)
    print(f"🎯 Supported Poses: {len(pose_analyzers)}")
    print(f"📊 Pose Types: {list(pose_analyzers.keys())}")
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