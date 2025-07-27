"""
Session management models and database functions for Yoga Assistant API
"""

from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import json
from datetime import datetime, timedelta

# Session Management Pydantic models
class PoseDetail(BaseModel):
    pose_name: str
    accuracy: float
    feedback: Optional[List[str]] = None
    improvements: Optional[List[str]] = None

class SessionCreate(BaseModel):
    duration: int  # in minutes
    poses: List[PoseDetail]
    avg_accuracy: float

class SessionPoseResponse(BaseModel):
    pose_name: str
    accuracy: float
    feedback: Optional[List[str]] = None
    improvements: Optional[List[str]] = None
    timestamp: str

class SessionResponse(BaseModel):
    id: int
    date: str
    duration: int
    poses: List[SessionPoseResponse]
    avg_accuracy: float
    total_poses: int
    user_id: int

class SessionsResponse(BaseModel):
    success: bool
    data: List[SessionResponse]
    message: Optional[str] = None

class SessionCreateResponse(BaseModel):
    success: bool
    data: SessionResponse
    message: Optional[str] = None

class DashboardStats(BaseModel):
    sessions_this_week: int
    mindful_minutes: int
    current_streak: int
    total_avg_accuracy: float

class StatsResponse(BaseModel):
    success: bool
    data: DashboardStats
    message: Optional[str] = None

def get_sessions_stats(user_id: int):
    """Get dashboard statistics for a user"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Get current week start (Monday)
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    
    # Sessions this week
    cursor.execute('''
        SELECT COUNT(*) FROM sessions 
        WHERE user_id = ? AND date >= ?
    ''', (user_id, week_start.strftime('%Y-%m-%d')))
    sessions_this_week = cursor.fetchone()[0]
    
    # Total mindful minutes
    cursor.execute('''
        SELECT SUM(duration) FROM sessions WHERE user_id = ?
    ''', (user_id,))
    result = cursor.fetchone()
    total_mindful_minutes = result[0] if result[0] else 0
    
    # Current streak calculation
    cursor.execute('''
        SELECT date FROM sessions 
        WHERE user_id = ? 
        ORDER BY date DESC
    ''', (user_id,))
    session_dates = cursor.fetchall()
    
    current_streak = 0
    if session_dates:
        current_date = datetime.now().date()
        consecutive_days = 0
        
        for session_date in session_dates:
            session_date_obj = datetime.strptime(session_date[0], '%Y-%m-%d %H:%M:%S').date()
            if session_date_obj == current_date - timedelta(days=consecutive_days):
                consecutive_days += 1
            else:
                break
        current_streak = consecutive_days
    
    # Total average accuracy
    cursor.execute('''
        SELECT AVG(avg_accuracy) FROM sessions WHERE user_id = ?
    ''', (user_id,))
    result = cursor.fetchone()
    total_avg_accuracy = result[0] if result[0] else 0
    
    conn.close()
    
    return {
        "sessions_this_week": sessions_this_week,
        "mindful_minutes": total_mindful_minutes,
        "current_streak": current_streak,
        "total_avg_accuracy": round(total_avg_accuracy, 1)
    }

def get_user_sessions(user_id: int):
    """Get all sessions for a user"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Get sessions
    cursor.execute('''
        SELECT id, date, duration, avg_accuracy, total_poses 
        FROM sessions 
        WHERE user_id = ? 
        ORDER BY date DESC
    ''', (user_id,))
    sessions = cursor.fetchall()
    
    session_responses = []
    for session in sessions:
        # Get poses for this session
        cursor.execute('''
            SELECT pose_name, accuracy, feedback, improvements, timestamp 
            FROM session_poses 
            WHERE session_id = ?
            ORDER BY timestamp
        ''', (session[0],))
        poses = cursor.fetchall()
        
        pose_responses = []
        for pose in poses:
            pose_responses.append(SessionPoseResponse(
                pose_name=pose[0],
                accuracy=pose[1],
                feedback=json.loads(pose[2]) if pose[2] else None,
                improvements=json.loads(pose[3]) if pose[3] else None,
                timestamp=pose[4]
            ))
        
        session_responses.append(SessionResponse(
            id=session[0],
            date=session[1],
            duration=session[2],
            poses=pose_responses,
            avg_accuracy=session[3],
            total_poses=session[4],
            user_id=user_id
        ))
    
    conn.close()
    return session_responses

def create_session(user_id: int, session_data: SessionCreate):
    """Create a new session for a user"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Create session
    cursor.execute('''
        INSERT INTO sessions (user_id, duration, avg_accuracy, total_poses)
        VALUES (?, ?, ?, ?)
    ''', (user_id, session_data.duration, session_data.avg_accuracy, len(session_data.poses)))
    
    session_id = cursor.lastrowid
    
    # Create session poses
    for pose_data in session_data.poses:
        cursor.execute('''
            INSERT INTO session_poses (session_id, pose_name, accuracy, feedback, improvements)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            pose_data.pose_name,
            pose_data.accuracy,
            json.dumps(pose_data.feedback) if pose_data.feedback else None,
            json.dumps(pose_data.improvements) if pose_data.improvements else None
        ))
    
    conn.commit()
    
    # Get the created session with poses
    cursor.execute('''
        SELECT id, date, duration, avg_accuracy, total_poses 
        FROM sessions 
        WHERE id = ?
    ''', (session_id,))
    session = cursor.fetchone()
    
    cursor.execute('''
        SELECT pose_name, accuracy, feedback, improvements, timestamp 
        FROM session_poses 
        WHERE session_id = ?
        ORDER BY timestamp
    ''', (session_id,))
    poses = cursor.fetchall()
    
    pose_responses = []
    for pose in poses:
        pose_responses.append(SessionPoseResponse(
            pose_name=pose[0],
            accuracy=pose[1],
            feedback=json.loads(pose[2]) if pose[2] else None,
            improvements=json.loads(pose[3]) if pose[3] else None,
            timestamp=pose[4]
        ))
    
    session_response = SessionResponse(
        id=session[0],
        date=session[1],
        duration=session[2],
        poses=pose_responses,
        avg_accuracy=session[3],
        total_poses=session[4],
        user_id=user_id
    )
    
    conn.close()
    return session_response

def get_session_by_id(session_id: int, user_id: int):
    """Get a specific session by ID"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Get session
    cursor.execute('''
        SELECT id, date, duration, avg_accuracy, total_poses, user_id 
        FROM sessions 
        WHERE id = ?
    ''', (session_id,))
    session = cursor.fetchone()
    
    if not session:
        conn.close()
        return None
    
    if session[5] != user_id:  # user_id is at index 5
        conn.close()
        return None
    
    # Get poses for this session
    cursor.execute('''
        SELECT pose_name, accuracy, feedback, improvements, timestamp 
        FROM session_poses 
        WHERE session_id = ?
        ORDER BY timestamp
    ''', (session_id,))
    poses = cursor.fetchall()
    
    pose_responses = []
    for pose in poses:
        pose_responses.append(SessionPoseResponse(
            pose_name=pose[0],
            accuracy=pose[1],
            feedback=json.loads(pose[2]) if pose[2] else None,
            improvements=json.loads(pose[3]) if pose[3] else None,
            timestamp=pose[4]
        ))
    
    session_response = SessionResponse(
        id=session[0],
        date=session[1],
        duration=session[2],
        poses=pose_responses,
        avg_accuracy=session[3],
        total_poses=session[4],
        user_id=session[5]
    )
    
    conn.close()
    return session_response

def update_session(session_id: int, user_id: int, session_data: SessionCreate):
    """Update an existing session"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Check if session exists and belongs to user
    cursor.execute('''
        SELECT user_id FROM sessions WHERE id = ?
    ''', (session_id,))
    session = cursor.fetchone()
    
    if not session or session[0] != user_id:
        conn.close()
        return None
    
    # Update session
    cursor.execute('''
        UPDATE sessions 
        SET duration = ?, avg_accuracy = ?, total_poses = ?
        WHERE id = ?
    ''', (session_data.duration, session_data.avg_accuracy, len(session_data.poses), session_id))
    
    # Delete existing poses
    cursor.execute('DELETE FROM session_poses WHERE session_id = ?', (session_id,))
    
    # Create new poses
    for pose_data in session_data.poses:
        cursor.execute('''
            INSERT INTO session_poses (session_id, pose_name, accuracy, feedback, improvements)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            pose_data.pose_name,
            pose_data.accuracy,
            json.dumps(pose_data.feedback) if pose_data.feedback else None,
            json.dumps(pose_data.improvements) if pose_data.improvements else None
        ))
    
    conn.commit()
    
    # Get updated session
    cursor.execute('''
        SELECT id, date, duration, avg_accuracy, total_poses 
        FROM sessions 
        WHERE id = ?
    ''', (session_id,))
    session = cursor.fetchone()
    
    cursor.execute('''
        SELECT pose_name, accuracy, feedback, improvements, timestamp 
        FROM session_poses 
        WHERE session_id = ?
        ORDER BY timestamp
    ''', (session_id,))
    poses = cursor.fetchall()
    
    pose_responses = []
    for pose in poses:
        pose_responses.append(SessionPoseResponse(
            pose_name=pose[0],
            accuracy=pose[1],
            feedback=json.loads(pose[2]) if pose[2] else None,
            improvements=json.loads(pose[3]) if pose[3] else None,
            timestamp=pose[4]
        ))
    
    session_response = SessionResponse(
        id=session[0],
        date=session[1],
        duration=session[2],
        poses=pose_responses,
        avg_accuracy=session[3],
        total_poses=session[4],
        user_id=user_id
    )
    
    conn.close()
    return session_response

def delete_session(session_id: int, user_id: int):
    """Delete a session"""
    conn = sqlite3.connect('yoga_assistant.db')
    cursor = conn.cursor()
    
    # Check if session exists and belongs to user
    cursor.execute('''
        SELECT user_id FROM sessions WHERE id = ?
    ''', (session_id,))
    session = cursor.fetchone()
    
    if not session or session[0] != user_id:
        conn.close()
        return False
    
    # Delete session (poses will be deleted automatically due to CASCADE)
    cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()
    
    return True 