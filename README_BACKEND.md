# Yoga Assistant Backend - Session Management

This document describes the backend implementation for the Yoga Assistant system with session management functionality.

## 🚀 Features

### New in v2.0
- **User Authentication**: JWT-based login/signup system
- **Session Management**: Track yoga practice sessions with pose details
- **Dashboard Statistics**: Real-time statistics and progress tracking
- **Database Integration**: SQLite database with session tracking
- **API Documentation**: Auto-generated OpenAPI documentation

### Existing Features
- **Pose Analysis**: AI-powered yoga pose detection
- **Multiple Poses**: Support for 11 different yoga poses
- **Enhanced Analyzer**: Hybrid ML + rule-based detection
- **Real-time Feedback**: Instant pose analysis and feedback

## 📁 Project Structure

```
yoga-assistant/
├── main.py              # FastAPI application with all endpoints
├── session_models.py    # Session management models and functions
├── test_backend.py      # Test script for backend functionality
├── requirements.txt     # Python dependencies
├── pose_analyzers/     # Pose analysis modules
├── utils/              # Utility modules
└── README_BACKEND.md   # This file
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
cd yoga-assistant
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the `yoga-assistant` directory:

```env
# Security Configuration
SECRET_KEY=your-secret-key-change-in-production
# Generate a secure key: python -c "import secrets; print(secrets.token_urlsafe(32))"

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS Configuration (for production)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=info
```

### 3. Run the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

## 📊 Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    sex TEXT NOT NULL,
    dob TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Sessions Table
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration INTEGER NOT NULL,
    avg_accuracy REAL NOT NULL,
    total_poses INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);
```

### Session Poses Table
```sql
CREATE TABLE session_poses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    pose_name TEXT NOT NULL,
    accuracy REAL NOT NULL,
    feedback TEXT,
    improvements TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);
```

## 🔌 API Endpoints

### Authentication
- `POST /auth/signup` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user info
- `GET /auth/protected` - Protected route example

### Session Management
- `GET /api/sessions/stats` - Dashboard statistics
- `GET /api/sessions` - Get all user sessions
- `POST /api/sessions` - Create new session
- `GET /api/sessions/{session_id}` - Get specific session
- `PUT /api/sessions/{session_id}` - Update session
- `DELETE /api/sessions/{session_id}` - Delete session

### Pose Analysis
- `GET /api/poses` - Get supported poses
- `GET /api/poses/{pose_name}/info` - Get pose information
- `POST /api/analyze/{pose_name}` - Analyze specific pose
- `POST /api/analyze-pose` - Enhanced pose analysis
- `POST /api/batch-analyze` - Batch pose analysis

### System
- `GET /` - API information
- `GET /api/health` - Health check
- `GET /api/enhanced-pose-info` - Enhanced analyzer info
- `GET /api/analytics/summary` - Analytics summary

## 📝 Usage Examples

### 1. User Registration
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "fullname": "John Doe",
    "email": "john@example.com",
    "password": "password123",
    "sex": "male",
    "dob": "1990-01-01"
  }'
```

### 2. User Login
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "password123"
  }'
```

### 3. Get Dashboard Statistics
```bash
curl -X GET "http://localhost:8000/api/sessions/stats" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 4. Create Session
```bash
curl -X POST "http://localhost:8000/api/sessions" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 25,
    "poses": [
      {
        "pose_name": "Mountain Pose",
        "accuracy": 92.5,
        "feedback": ["Good alignment", "Keep shoulders relaxed"],
        "improvements": ["Lengthen spine more"]
      }
    ],
    "avg_accuracy": 92.5
  }'
```

### 5. Get Sessions
```bash
curl -X GET "http://localhost:8000/api/sessions" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🧪 Testing

### Run the Test Script

```bash
python test_backend.py
```

This will test:
1. Health check
2. User signup
3. User login
4. Dashboard statistics
5. Session creation
6. Session retrieval
7. Statistics updates

### Manual Testing

You can also test the API manually using the Swagger UI:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt password hashing
- **User Authorization**: Session access control
- **Input Validation**: Pydantic model validation
- **CORS Support**: Cross-origin resource sharing

## 📊 Dashboard Statistics

The backend calculates the following statistics:

- **Sessions this week**: Number of sessions in the current week
- **Mindful minutes**: Total practice time across all sessions
- **Current streak**: Consecutive days with practice sessions
- **Total average accuracy**: Average accuracy across all sessions

## 🔄 Session Management

### Session Creation
When a user completes a yoga session:
1. Session data is sent to `/api/sessions`
2. Session is stored with user ID, duration, and average accuracy
3. Individual pose details are stored with feedback and improvements
4. Dashboard statistics are automatically updated

### Session Retrieval
- Users can view all their sessions
- Each session shows pose details with accuracy and feedback
- Sessions are ordered by date (most recent first)

## 🚀 Production Deployment

### 1. Environment Variables
- Set `SECRET_KEY` to a secure random string
- Configure `ALLOWED_ORIGINS` for your frontend domain

### 2. Database
- Consider using PostgreSQL for production
- Set up proper database backups
- Configure connection pooling

### 3. Security
- Use HTTPS in production
- Implement rate limiting
- Set up proper CORS configuration
- Use environment variables for sensitive data

### 4. Monitoring
- Set up logging
- Monitor API performance
- Set up error tracking

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check if the database file exists
   - Ensure write permissions in the directory

2. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Authentication Errors**
   - Verify `SECRET_KEY` is set
   - Check token format in Authorization header

4. **CORS Errors**
   - Configure `ALLOWED_ORIGINS` in environment
   - Check frontend URL is included

### Logs
Check the console output for detailed error messages and API logs.

## 📚 API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 Frontend Integration

The frontend is configured to connect to this backend:
- API base URL: `http://localhost:8000`
- Authentication endpoints: `/auth/*`
- Session endpoints: `/api/sessions/*`
- Pose analysis endpoints: `/api/analyze/*`

## 📈 Future Enhancements

- **Database Migrations**: Alembic for schema changes
- **Caching**: Redis for performance optimization
- **File Upload**: Support for video/image uploads
- **Real-time Updates**: WebSocket support
- **Analytics**: Advanced progress tracking
- **Social Features**: User sharing and leaderboards

## 🤝 Support

For issues or questions:
1. Check the API documentation at `/docs`
2. Review the logs for error details
3. Ensure all dependencies are installed
4. Verify environment configuration
5. Run the test script to verify functionality

---

**🎉 The backend is now fully functional with session management!** 