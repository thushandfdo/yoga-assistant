# Yoga Assistant API

A comprehensive FastAPI backend for AI-powered yoga pose analysis and feedback. This system uses MediaPipe for pose detection and provides real-time analysis with personalized feedback for multiple yoga poses.

## 🧘‍♀️ Features

- **11 Supported Yoga Poses**: Chair, Mountain, Tree, Downward Dog, Goddess, Lord of the Dance, Low Lunge, Side Plank, Staff, T-Pose, and Warrior 2
- **AI-Powered Analysis**: Advanced pose detection using MediaPipe and custom ML models
- **Real-time Feedback**: Detailed analysis with specific improvement suggestions
- **Skill Level Adaptation**: Customized feedback based on user experience level
- **RESTful API**: Clean, well-documented endpoints with OpenAPI specification
- **Batch Processing**: Analyze multiple poses simultaneously
- **Comprehensive Metrics**: Joint angles, alignment scores, and balance analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yoga-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Access the API**
   - API Server: http://localhost:8000
   - Interactive Documentation: http://localhost:8000/docs
   - ReDoc Documentation: http://localhost:8000/redoc

## 📚 API Documentation

### Core Endpoints

#### Health Check
```http
GET /api/health
```
Returns API status and available poses.

#### Get All Supported Poses
```http
GET /api/poses
```
Returns list of all supported yoga poses with metadata.

#### Get Pose Information
```http
GET /api/poses/{pose_name}/info
```
Returns detailed information about a specific pose including benefits, instructions, and common mistakes.

#### Analyze Single Pose
```http
POST /api/analyze/{pose_name}
```
Upload a video file to analyze a specific yoga pose.

**Parameters:**
- `pose_name`: Name of the pose to analyze (chair, mountain, tree, etc.)
- `video`: Video file (MP4, WebM, AVI, MOV, MKV)
- `skill_level`: User's experience level (beginner, intermediate, advanced)

**Response Example:**
```json
{
  "success": true,
  "pose_detected": "Chair Pose",
  "score": 85,
  "feedback": [
    "Excellent! Your leg positioning is perfect.",
    "Try to raise your arms a bit higher overhead."
  ],
  "angles": {
    "right_elbow": 175,
    "left_elbow": 173,
    "right_knee": 95,
    "left_knee": 98
  },
  "analysis_timestamp": "2024-01-15 10:30:45",
  "skill_level": "beginner"
}
```

#### Batch Analysis
```http
POST /api/batch-analyze
```
Analyze multiple videos for different poses in a single request.

### Supported Poses

| Pose Name | Difficulty | Description |
|-----------|------------|-------------|
| `chair` | Intermediate | Chair Pose (Utkatasana) - Strengthens legs and core |
| `mountain` | Beginner | Mountain Pose (Tadasana) - Foundation standing pose |
| `tree` | Intermediate | Tree Pose (Vrksasana) - Balance and focus pose |
| `downdog` | Beginner | Downward Dog - Full body stretch and strength |
| `goddess` | Intermediate | Goddess Pose - Hip opening and leg strength |
| `lord` | Advanced | Lord of the Dance - Advanced balance pose |
| `lowlung` | Beginner | Low Lunge - Hip flexibility and leg strength |
| `side_plank` | Intermediate | Side Plank - Core and arm strength |
| `staff` | Beginner | Staff Pose - Seated posture and back strength |
| `t_pose` | Beginner | T-Pose - Basic standing pose with arms extended |
| `warrior2` | Intermediate | Warrior 2 - Standing strength and endurance |

## 🔧 Technical Architecture

### Project Structure
```
yoga-assistant/
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── setup.py                   # Project setup and installation script
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── pose_analyzers/            # Pose analysis modules
│   ├── __init__.py            # Package initialization
│   ├── base_analyzer.py       # Abstract base class for pose analyzers
│   ├── chair_pose.py          # Chair pose analyzer
│   ├── mountain_pose.py       # Mountain pose analyzer
│   ├── tree_pose.py           # Tree pose analyzer
│   ├── downdog_pose.py        # Downward Dog pose analyzer
│   ├── goddess_pose.py        # Goddess pose analyzer
│   ├── lord_pose.py           # Lord of the Dance pose analyzer
│   ├── lowlung_pose.py        # Low Lunge pose analyzer
│   ├── side_plank.py          # Side Plank pose analyzer
│   ├── staff_pose.py          # Staff pose analyzer
│   ├── t_pose.py              # T-Pose analyzer
│   └── warrior2_pose.py       # Warrior 2 pose analyzer
└── utils/                     # Utility functions
    ├── __init__.py            # Package initialization
    ├── video_processor.py     # Video frame extraction utilities
    └── angle_calculator.py    # Angle calculation utilities
```

### Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **MediaPipe**: Google's ML framework for pose detection
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing for angle calculations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning for advanced pose classification
- **Uvicorn**: ASGI server for serving the application
- **Python-multipart**: File upload handling
- **Aiofiles**: Async file operations
- **Pillow**: Image processing
- **Python-jose**: JWT token handling
- **Passlib**: Password hashing

### Pose Analysis Pipeline

1. **Video Processing**: Extract frames from uploaded video files using `utils/video_processor.py`
2. **Pose Detection**: Use MediaPipe to detect body landmarks
3. **Feature Extraction**: Calculate joint angles and body alignment metrics using `utils/angle_calculator.py`
4. **Classification**: Determine if pose matches target pose pattern using specific pose analyzers
5. **Feedback Generation**: Provide specific improvement suggestions
6. **Response Formatting**: Return structured JSON with analysis results

## 🎯 Usage Examples

### Using curl

```bash
# Analyze a Chair Pose video
curl -X POST "http://localhost:8000/api/analyze/chair" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@my_pose_video.mp4" \
     -F "skill_level=beginner"

# Get information about Tree Pose
curl "http://localhost:8000/api/poses/tree/info"
```

### Using Python requests

```python
import requests

# Analyze a pose
with open('pose_video.mp4', 'rb') as video_file:
    response = requests.post(
        'http://localhost:8000/api/analyze/mountain',
        files={'video': video_file},
        data={'skill_level': 'intermediate'}
    )
    
result = response.json()
print(f"Pose: {result['pose_detected']}")
print(f"Score: {result['score']}%")
print("Feedback:", result['feedback'])
```

## 🔄 Development

### Adding New Poses

1. Create a new analyzer class inheriting from `BasePoseAnalyzer` in `pose_analyzers/base_analyzer.py`
2. Implement required methods: `get_pose_name()`, `get_description()`, `get_difficulty()`, `analyze_pose()`
3. Add the analyzer to `pose_analyzers/__init__.py`
4. Update the pose mapping in `main.py`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 🔒 Security Considerations

- Input validation for uploaded files
- File size limits to prevent abuse
- Temporary file cleanup after processing
- CORS configuration for production deployment

## 📦 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Allowed CORS origins
- `MAX_FILE_SIZE`: Maximum upload file size

## 🐛 Troubleshooting

### Common Issues

1. **MediaPipe Installation**: Ensure compatible Python version and system dependencies
2. **Video Format**: Supported formats are MP4, WebM, AVI, MOV, MKV
3. **Memory Usage**: Large video files may require more RAM for processing
4. **Pose Detection**: Ensure good lighting and full body visibility in videos

### Performance Optimization

- Use smaller video files or extract single frames
- Implement video compression before analysis
- Add caching for frequently analyzed poses
- Use async processing for batch operations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support and questions:
- Check the [API documentation](http://localhost:8000/docs)
- Review common issues in this README
- Submit issues on the repository

## 🚀 Future Enhancements

- [ ] Advanced ML models for each specific pose
- [ ] Real-time video streaming analysis
- [ ] Progress tracking and analytics
- [ ] Integration with fitness tracking apps
- [ ] Multi-person pose analysis
- [ ] 3D pose reconstruction
- [ ] Voice feedback generation
- [ ] Mobile app companion

---

Built with ❤️ for the yoga community. Namaste! 🙏 