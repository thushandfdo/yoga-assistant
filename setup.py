#!/usr/bin/env python3
"""
Setup script for Yoga Assistant API
Helps with installation and running the application
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_system_requirements():
    """Check system-specific requirements"""
    system = platform.system()
    print(f"\n🖥️  System: {system}")
    
    if system == "Darwin":  # macOS
        print("✅ macOS detected - MediaPipe should work well")
    elif system == "Linux":
        print("✅ Linux detected - MediaPipe should work well")
    elif system == "Windows":
        print("✅ Windows detected - Make sure Visual C++ redistributables are installed")
    
    return True

def test_imports():
    """Test if all critical imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI not found")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn")
    except ImportError:
        print("❌ Uvicorn not found")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe {mp.__version__}")
    except ImportError:
        print("❌ MediaPipe not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    return True

def run_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Yoga Assistant API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

def main():
    """Main setup function"""
    print("🧘‍♀️ Yoga Assistant API Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Ask user if they want to install dependencies
    while True:
        install = input("\n📦 Install dependencies? (y/n): ").lower().strip()
        if install in ['y', 'yes']:
            if not install_dependencies():
                sys.exit(1)
            break
        elif install in ['n', 'no']:
            print("⚠️  Skipping dependency installation")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # Test imports
    if not test_imports():
        print("\n❌ Some dependencies are missing. Please install them first.")
        sys.exit(1)
    
    # Ask if user wants to start the server
    while True:
        start = input("\n🚀 Start the API server? (y/n): ").lower().strip()
        if start in ['y', 'yes']:
            run_server()
            break
        elif start in ['n', 'no']:
            print("\n✅ Setup complete! Run 'python main.py' to start the server.")
            break
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main() 