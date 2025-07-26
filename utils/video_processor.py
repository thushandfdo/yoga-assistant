"""
Video Processing Utilities
Functions for extracting and processing video frames
"""

import cv2
import numpy as np
from typing import Optional


def extract_frame_from_video(video_path: str, frame_time: float = 3.0) -> Optional[np.ndarray]:
    """
    Extract a frame from video file at specified time or middle frame
    
    Args:
        video_path: Path to the video file
        frame_time: Time in seconds to extract frame (default: 3.0)
        
    Returns:
        Extracted frame as numpy array or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Determine which frame to extract
        if fps > 0 and duration > frame_time:
            # Extract frame at specified time
            frame_number = int(fps * frame_time)
        else:
            # Use middle frame if video is shorter than frame_time
            frame_number = total_frames // 2
        
        # Ensure frame number is within bounds
        frame_number = max(0, min(frame_number, total_frames - 1))
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret and frame is not None:
            return frame
        else:
            print(f"Error: Could not read frame {frame_number} from video")
            return None
            
    except Exception as e:
        print(f"Error extracting frame from video: {e}")
        return None


def extract_multiple_frames(video_path: str, num_frames: int = 5) -> list:
    """
    Extract multiple frames evenly distributed throughout the video
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        
    Returns:
        List of extracted frames as numpy arrays
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            # If video has fewer frames than requested, extract all available
            frame_indices = list(range(total_frames))
        else:
            # Extract frames evenly distributed
            step = total_frames // num_frames
            frame_indices = [i * step for i in range(num_frames)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frames.append(frame)
        
        cap.release()
        
    except Exception as e:
        print(f"Error extracting multiple frames: {e}")
    
    return frames


def validate_video_file(video_path: str) -> dict:
    """
    Validate video file and return properties
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with validation results and video properties
    """
    result = {
        'valid': False,
        'error': None,
        'properties': {}
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            result['error'] = "Could not open video file"
            return result
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Validate properties
        if fps <= 0:
            result['error'] = "Invalid frame rate"
        elif total_frames <= 0:
            result['error'] = "No frames found in video"
        elif width <= 0 or height <= 0:
            result['error'] = "Invalid video dimensions"
        elif duration < 0.5:  # Minimum 0.5 seconds
            result['error'] = "Video too short (minimum 0.5 seconds)"
        else:
            result['valid'] = True
            result['properties'] = {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'format': 'Unknown'  # Could be enhanced to detect format
            }
        
        cap.release()
        
    except Exception as e:
        result['error'] = f"Error validating video: {str(e)}"
    
    return result


def resize_frame(frame: np.ndarray, target_width: int = 640, target_height: int = 480) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame as numpy array
        target_width: Target width in pixels
        target_height: Target height in pixels
        
    Returns:
        Resized frame
    """
    try:
        height, width = frame.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        scale_width = target_width / width
        scale_height = target_height / height
        scale = min(scale_width, scale_height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized_frame
        
    except Exception as e:
        print(f"Error resizing frame: {e}")
        return frame 