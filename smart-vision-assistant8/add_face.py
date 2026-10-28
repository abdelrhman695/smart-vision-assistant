"""
Helper script to add a new face to the system
يساعدك في إضافة وجه جديد للنظام
"""

import cv2
import os
import sys
from pathlib import Path
import face_recognition
import numpy as np

def capture_face(name: str, known_faces_dir: str = "known_faces"):
    """
    Capture a face from camera and save it
    التقاط وجه من الكاميرا وحفظه
    """
    # Create directory if it doesn't exist
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera / لا يمكن فتح الكاميرا")
        return False
    
    print(f"📹 Camera opened. Position your face and press SPACE to capture / افتح الكاميرا ووضع وجهك واضغط مسافة للالتقاط")
    print("Press ESC to cancel / اضغط ESC للإلغاء")
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, f"Adding: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE: Capture | ESC: Cancel", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Add Face - إضافة وجه', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            if len(faces) == 0:
                print("⚠️ No face detected! Please position your face properly / لم يتم اكتشاف وجه! ضع وجهك بشكل صحيح")
                continue
            elif len(faces) > 1:
                print("⚠️ Multiple faces detected! Please ensure only one face is visible / تم اكتشاف عدة وجوه! تأكد من ظهور وجه واحد فقط")
                continue
            
            # Save the image
            filename = f"{known_faces_dir}/{name.replace(' ', '_').lower()}.jpg"
            cv2.imwrite(filename, frame)
            
            # Verify the face can be encoded
            try:
                image = face_recognition.load_image_file(filename)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    print(f"✅ Face captured and saved: {filename}")
                    print(f"✅ تم التقاط الوجه وحفظه: {filename}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("❌ Could not encode face, please try again / لا يمكن ترميز الوجه، حاول مرة أخرى")
                    os.remove(filename)
            except Exception as e:
                print(f"❌ Error processing face: {e}")
                if os.path.exists(filename):
                    os.remove(filename)
                    
        elif key == 27:  # ESC to cancel
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def add_face_from_file(image_path: str, name: str, known_faces_dir: str = "known_faces"):
    """
    Add a face from an existing image file
    إضافة وجه من ملف صورة موجود
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        # Load and verify the image
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            print(f"❌ No face found in {image_path}")
            return False
        
        if len(face_encodings) > 1:
            print(f"⚠️ Multiple faces found in {image_path}, using the first one")
        
        # Create directory if it doesn't exist
        os.makedirs(known_faces_dir, exist_ok=True)
        
        # Copy image to known_faces directory
        filename = f"{known_faces_dir}/{name.replace(' ', '_').lower()}.jpg"
        
        # Load with OpenCV and save
        cv_image = cv2.imread(image_path)
        cv2.imwrite(filename, cv_image)
        
        print(f"✅ Face added: {filename}")
        print(f"✅ تم إضافة الوجه: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

def list_known_faces(known_faces_dir: str = "known_faces"):
    """List all known faces"""
    if not os.path.exists(known_faces_dir):
        print("📁 No known_faces directory found")
        return
    
    face_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        face_files.extend(Path(known_faces_dir).glob(ext))
    
    if not face_files:
        print("📷 No face images found")
        return
    
    print(f"👥 Known faces ({len(face_files)}):")
    for face_file in face_files:
        name = face_file.stem.replace('_', ' ').title()
        print(f"  - {name} ({face_file.name})")

def main():
    """Main function"""
    print("🎭 Face Management System / نظام إدارة الوجوه")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage / الاستخدام:")
        print("  python add_face.py capture <name>     - Capture from camera / التقاط من الكاميرا")
        print("  python add_face.py add <image_path> <name>  - Add from file / إضافة من ملف")
        print("  python add_face.py list               - List known faces / عرض الوجوه المعروفة")
        print()
        print("Examples / أمثلة:")
        print("  python add_face.py capture \"Ahmed Mohamed\"")
        print("  python add_face.py add photo.jpg \"Sara Ali\"")
        print("  python add_face.py list")
        return
    
    command = sys.argv[1].lower()
    
    if command == "capture":
        if len(sys.argv) < 3:
            print("❌ Please provide a name / يرجى تقديم اسم")
            return
        
        name = sys.argv[2]
        print(f"📸 Capturing face for: {name}")
        
        if capture_face(name):
            print("\n🎉 Success! Now restart the vision system to use the new face.")
            print("🎉 نجح! الآن أعد تشغيل نظام الرؤية لاستخدام الوجه الجديد.")
        else:
            print("\n❌ Failed to capture face.")
            print("❌ فشل في التقاط الوجه.")
    
    elif command == "add":
        if len(sys.argv) < 4:
            print("❌ Please provide image path and name / يرجى تقديم مسار الصورة والاسم")
            return
        
        image_path = sys.argv[2]
        name = sys.argv[3]
        
        print(f"📁 Adding face from {image_path} for: {name}")
        
        if add_face_from_file(image_path, name):
            print("\n🎉 Success! Now restart the vision system to use the new face.")
            print("🎉 نجح! الآن أعد تشغيل نظام الرؤية لاستخدام الوجه الجديد.")
        else:
            print("\n❌ Failed to add face.")
            print("❌ فشل في إضافة الوجه.")
    
    elif command == "list":
        list_known_faces()
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
