"""
Setup script for Smart Vision System
Handles installation and environment setup
"""

import subprocess
import sys
import os
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logging.error("❌ Python 3.8+ required. Current version: {}.{}.{}".format(
            version.major, version.minor, version.micro))
        return False
    
    logging.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    logging.info(f"🖥️ Detected system: {system}")
    
    if system == "linux":
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-dev",
            "sudo apt-get install -y cmake",
            "sudo apt-get install -y build-essential",
            "sudo apt-get install -y libopencv-dev",
            "sudo apt-get install -y portaudio19-dev",
            "sudo apt-get install -y ffmpeg"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd.split(), check=True)
                logging.info(f"✅ {cmd}")
            except subprocess.CalledProcessError:
                logging.warning(f"⚠️ Failed: {cmd}")
    
    elif system == "darwin":  # macOS
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            commands = [
                "brew install cmake",
                "brew install portaudio",
                "brew install opencv",
                "brew install ffmpeg"
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd.split(), check=True)
                    logging.info(f"✅ {cmd}")
                except subprocess.CalledProcessError:
                    logging.warning(f"⚠️ Failed: {cmd}")
        except subprocess.CalledProcessError:
            logging.warning("Homebrew not found. Please install Homebrew first.")
    
    elif system == "windows":
        logging.info("Windows: Please ensure Visual Studio Build Tools are installed")
        logging.info("Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

def install_python_packages():
    """Install Python packages with compatibility checks"""
    logging.info("📦 Installing Python packages...")
    
    # Core packages first
    core_packages = [
        "numpy==1.24.3",
        "opencv-python==4.8.1.78",
        "Pillow==10.0.1"
    ]
    
    # AI/ML packages
    ai_packages = [
        "torch==2.0.1",
        "torchvision==0.15.2",
        "ultralytics==8.0.196",
        "mediapipe==0.10.7"
    ]
    
    # Face recognition (can be problematic)
    face_packages = [
        "dlib==19.24.2",
        "face-recognition==1.3.0"
    ]
    
    # Voice packages (optional)
    voice_packages = [
        "SpeechRecognition==3.10.0",
        "pyttsx3==2.90"
    ]
    
    # Utility packages
    util_packages = [
        "tqdm==4.66.1",
        "requests==2.31.0",
        "pandas==2.0.3",
        "scipy==1.11.2",
        "psutil==5.9.5",
        "pyyaml==6.0.1"
    ]
    
    package_groups = [
        ("Core packages", core_packages),
        ("AI/ML packages", ai_packages),
        ("Face recognition", face_packages),
        ("Voice packages", voice_packages),
        ("Utility packages", util_packages)
    ]
    
    for group_name, packages in package_groups:
        logging.info(f"📦 Installing {group_name}...")
        
        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                logging.info(f"✅ {package}")
            except subprocess.CalledProcessError as e:
                logging.error(f"❌ Failed to install {package}")
                
                # Special handling for problematic packages
                if "dlib" in package or "face-recognition" in package:
                    logging.info("💡 Face recognition packages failed - system will work with basic face detection")
                elif "pyaudio" in package:
                    logging.info("💡 Voice packages failed - voice features will be disabled")
                else:
                    logging.warning(f"⚠️ {package} failed - some features may not work")

def verify_installation():
    """Verify that key packages are working"""
    logging.info("🔍 Verifying installation...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("ultralytics", "YOLOv8"),
        ("mediapipe", "MediaPipe"),
        ("PIL", "Pillow")
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            logging.info(f"✅ {name}")
            success_count += 1
        except ImportError:
            logging.error(f"❌ {name}")
    
    # Optional imports
    optional_imports = [
        ("face_recognition", "Face Recognition"),
        ("speech_recognition", "Speech Recognition"),
        ("pyttsx3", "Text-to-Speech")
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            logging.info(f"✅ {name} (optional)")
        except ImportError:
            logging.warning(f"⚠️ {name} (optional) - not available")
    
    if success_count >= 5:
        logging.info("🎉 Installation successful!")
        return True
    else:
        logging.error("❌ Installation failed - too many core packages missing")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "known_faces",
        "models", 
        "logs",
        "screenshots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"📁 Created directory: {directory}")

def create_sample_files():
    """Create sample configuration and instruction files"""
    
    # Create known_faces README
    readme_content = """Smart Vision System - Known Faces

Add photos of people you want the system to recognize:

1. Supported formats: .jpg, .jpeg, .png
2. Name files with person's name: john_doe.jpg
3. Use clear, front-facing photos
4. One person per image
5. Good lighting is important

Examples:
- john_doe.jpg
- jane_smith.png  
- alice_johnson.jpeg

After adding photos, restart the system to load the new faces.
"""
    
    with open("known_faces/README.txt", "w") as f:
        f.write(readme_content)
    
    logging.info("📝 Created known_faces/README.txt")

def main():
    """Main setup function"""
    print("🚀 Smart Vision System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create directories
    create_directories()
    
    # Install Python packages
    install_python_packages()
    
    # Verify installation
    success = verify_installation()
    
    # Create sample files
    create_sample_files()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Add photos to known_faces/ directory")
        print("2. Run: python vision_system.py")
        print("3. Use keyboard controls:")
        print("   - Q: Quit")
        print("   - SPACE: Toggle detection")
        print("   - S: Save screenshot")
        print("   - D: Describe scene")
    else:
        print("\n❌ Setup had issues. Check the logs above.")
    
    return success

if __name__ == "__main__":
    main()
