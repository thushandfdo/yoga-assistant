#!/usr/bin/env python3
"""
Test script for Yoga Assistant Backend Session Management
"""

import requests
import json
import time

# Backend URL
BASE_URL = "http://localhost:8000"

def test_backend():
    """Test the backend session management functionality"""
    
    print("🧘‍♀️ Testing Yoga Assistant Backend Session Management")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: User signup
    print("\n2. Testing user signup...")
    signup_data = {
        "fullname": "Test User",
        "email": "test@example.com",
        "password": "password123",
        "sex": "male",
        "dob": "1990-01-01"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        if response.status_code == 200:
            print("✅ User signup successful")
            token_data = response.json()
            access_token = token_data['access_token']
            user_id = token_data['user_id']
            print(f"   User ID: {user_id}")
            print(f"   Token: {access_token[:20]}...")
        else:
            print(f"❌ User signup failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ User signup error: {e}")
        return False
    
    # Test 3: User login
    print("\n3. Testing user login...")
    login_data = {
        "email": "test@example.com",
        "password": "password123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if response.status_code == 200:
            print("✅ User login successful")
            token_data = response.json()
            access_token = token_data['access_token']
            print(f"   Token: {access_token[:20]}...")
        else:
            print(f"❌ User login failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ User login error: {e}")
        return False
    
    # Test 4: Get dashboard stats
    print("\n4. Testing dashboard stats...")
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/api/sessions/stats", headers=headers)
        if response.status_code == 200:
            print("✅ Dashboard stats retrieved")
            stats = response.json()
            print(f"   Sessions this week: {stats['data']['sessions_this_week']}")
            print(f"   Mindful minutes: {stats['data']['mindful_minutes']}")
            print(f"   Current streak: {stats['data']['current_streak']}")
            print(f"   Total avg accuracy: {stats['data']['total_avg_accuracy']}")
        else:
            print(f"❌ Dashboard stats failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Dashboard stats error: {e}")
        return False
    
    # Test 5: Get sessions (should be empty initially)
    print("\n5. Testing get sessions...")
    try:
        response = requests.get(f"{BASE_URL}/api/sessions", headers=headers)
        if response.status_code == 200:
            print("✅ Get sessions successful")
            sessions = response.json()
            print(f"   Number of sessions: {len(sessions['data'])}")
        else:
            print(f"❌ Get sessions failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Get sessions error: {e}")
        return False
    
    # Test 6: Create a session
    print("\n6. Testing create session...")
    session_data = {
        "duration": 25,
        "poses": [
            {
                "pose_name": "Mountain Pose",
                "accuracy": 92.5,
                "feedback": ["Good alignment", "Keep shoulders relaxed"],
                "improvements": ["Lengthen spine more"]
            },
            {
                "pose_name": "Tree Pose",
                "accuracy": 88.0,
                "feedback": ["Good balance", "Keep focus"],
                "improvements": ["Extend arms higher"]
            }
        ],
        "avg_accuracy": 90.25
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/sessions", json=session_data, headers=headers)
        if response.status_code == 200:
            print("✅ Create session successful")
            session = response.json()
            session_id = session['data']['id']
            print(f"   Session ID: {session_id}")
            print(f"   Duration: {session['data']['duration']} minutes")
            print(f"   Poses: {len(session['data']['poses'])}")
            print(f"   Avg accuracy: {session['data']['avg_accuracy']}%")
        else:
            print(f"❌ Create session failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Create session error: {e}")
        return False
    
    # Test 7: Get sessions again (should have one session)
    print("\n7. Testing get sessions after creation...")
    try:
        response = requests.get(f"{BASE_URL}/api/sessions", headers=headers)
        if response.status_code == 200:
            print("✅ Get sessions successful")
            sessions = response.json()
            print(f"   Number of sessions: {len(sessions['data'])}")
            if len(sessions['data']) > 0:
                session = sessions['data'][0]
                print(f"   Latest session: {session['id']}")
                print(f"   Duration: {session['duration']} minutes")
                print(f"   Poses: {len(session['poses'])}")
        else:
            print(f"❌ Get sessions failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Get sessions error: {e}")
        return False
    
    # Test 8: Get dashboard stats again (should have updated values)
    print("\n8. Testing dashboard stats after session creation...")
    try:
        response = requests.get(f"{BASE_URL}/api/sessions/stats", headers=headers)
        if response.status_code == 200:
            print("✅ Dashboard stats retrieved")
            stats = response.json()
            print(f"   Sessions this week: {stats['data']['sessions_this_week']}")
            print(f"   Mindful minutes: {stats['data']['mindful_minutes']}")
            print(f"   Current streak: {stats['data']['current_streak']}")
            print(f"   Total avg accuracy: {stats['data']['total_avg_accuracy']}")
        else:
            print(f"❌ Dashboard stats failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Dashboard stats error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Backend session management is working correctly.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    print("Starting backend tests...")
    print("Make sure the backend is running on http://localhost:8000")
    print("You can start it with: python main.py")
    print()
    
    success = test_backend()
    
    if success:
        print("\n✅ Backend session management is fully functional!")
    else:
        print("\n❌ Some tests failed. Please check the backend logs.") 