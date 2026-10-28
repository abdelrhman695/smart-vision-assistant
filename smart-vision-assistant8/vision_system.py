"""
Smart Vision System - Rebuilt with Compatible Libraries
Python 3.8+ compatible with carefully selected, stable libraries
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import mediapipe as mp
import face_recognition
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import threading
import queue
from typing import List, Dict, Tuple, Optional
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)

class SmartVisionSystem:
    """
    Complete Smart Vision System with compatible libraries
    Features: Object Detection, Face Recognition, Person Tracking, Voice Commands
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the Smart Vision System"""
        logging.info("🚀 Initializing Smart Vision System...")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # System state
        self.is_running = False
        self.detection_active = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Threading
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None
        
        # Initialize components
        self._init_camera()
        self._init_object_detector()
        self._init_face_system()
        self._init_voice_system()
        self._init_tracking()
        
        logging.info("✅ Smart Vision System initialized successfully!")
    
    def load_config(self, config_path: str) -> dict:
        """Load system configuration"""
        default_config = {
            "camera": {
                "index": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            "detection": {
                "object_confidence": 0.5,
                "face_confidence": 0.5,
                "face_recognition_tolerance": 0.6,
                "nms_threshold": 0.4
            },
            "paths": {
                "known_faces": "known_faces",
                "models": "models",
                "logs": "logs",
                "screenshots": "screenshots"
            },
            "display": {
                "show_fps": True,
                "show_confidence": True,
                "font_scale": 0.6,
                "box_thickness": 2
            },
            "voice": {
                "enabled": True,
                "language": "en-US",
                "commands": {
                    "start detection": "start",
                    "stop detection": "stop",
                    "take photo": "photo",
                    "describe scene": "describe",
                    "quit": "quit"
                }
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    self._deep_update(default_config, user_config)
            except Exception as e:
                logging.warning(f"Failed to load config: {e}, using defaults")
        else:
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logging.info(f"Created default config: {config_path}")
        
        # Create directories
        for path in default_config["paths"].values():
            os.makedirs(path, exist_ok=True)
        
        return default_config
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _init_camera(self):
        """Initialize camera system"""
        logging.info("📹 Initializing camera...")
        
        self.camera = None
        camera_config = self.config["camera"]
        
        try:
            self.camera = cv2.VideoCapture(camera_config["index"])
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_config['index']}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config["fps"])
            
            # Test frame capture
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Cannot capture frame from camera")
            
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            logging.info(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            logging.error(f"❌ Camera initialization failed: {e}")
            self.camera = None
    
    def _init_object_detector(self):
        """Initialize YOLO object detection"""
        logging.info("🎯 Initializing object detector...")
        
        try:
            # Use YOLOv8 nano for speed
            self.yolo_model = YOLO('yolov8n.pt')
            
            # Warm up the model
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.yolo_model(dummy_frame, verbose=False)
            
            logging.info("✅ YOLO object detector initialized")
            
        except Exception as e:
            logging.error(f"❌ Object detector initialization failed: {e}")
            self.yolo_model = None
    
    def _init_face_system(self):
        """Initialize MediaPipe face detection and recognition"""
        logging.info("👤 Initializing face system...")
        
        try:
            # Initialize MediaPipe Face Detection
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (2 meters)
                min_detection_confidence=self.config["detection"]["face_confidence"]
            )
            
            # Initialize MediaPipe Face Mesh for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Face recognition setup
            self.known_face_encodings = []
            self.known_face_names = []
            self.face_cache_file = Path(self.config["paths"]["known_faces"]) / "face_cache.pkl"
            
            self._load_known_faces()
            
            logging.info(f"✅ Face system initialized with {len(self.known_face_names)} known faces")
            
        except Exception as e:
            logging.error(f"❌ Face system initialization failed: {e}")
            self.face_detection = None
            self.face_mesh = None
    
    def _load_known_faces(self):
        """Load known faces from directory"""
        known_faces_dir = Path(self.config["paths"]["known_faces"])
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Try to load cached encodings first
        if self.face_cache_file.exists():
            try:
                with open(self.face_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
            # Check if cache is still valid (check file modification times)
                cache_time = datetime.fromisoformat(cached_data.get('timestamp', '1970-01-01'))
            
            # Check if any image file is newer than cache
                cache_valid = True
                face_files = list(known_faces_dir.glob("*.jpg")) + \
                            list(known_faces_dir.glob("*.jpeg")) + \
                            list(known_faces_dir.glob("*.png"))
            
                for face_file in face_files:
                    file_time = datetime.fromtimestamp(face_file.stat().st_mtime)
                    if file_time > cache_time:
                        cache_valid = False
                        break
            
                if cache_valid and cached_data.get('encodings') and cached_data.get('names'):
                    self.known_face_encodings = cached_data['encodings']
                    self.known_face_names = cached_data['names']
                    logging.info(f"📁 Loaded {len(self.known_face_names)} cached face encodings")
                    return
                else:
                    logging.info("🔄 Cache outdated, reloading faces...")
                
            except Exception as e:
                logging.warning(f"Failed to load face cache: {e}")

        # Load faces from images
        face_files = list(known_faces_dir.glob("*.jpg")) + \
                    list(known_faces_dir.glob("*.jpeg")) + \
                    list(known_faces_dir.glob("*.png"))
        
        if not face_files:
            logging.info("📷 No face images found in known_faces directory")
            self._create_face_instructions()
            return
        
        logging.info(f"🔍 Processing {len(face_files)} face images...")
        
        for face_file in face_files:
            try:
                logging.info(f"📸 Processing: {face_file.name}")
                
                # Load image using face_recognition (handles different formats better)
                image = face_recognition.load_image_file(str(face_file))
                
                # Find all faces in the image
                face_locations = face_recognition.face_locations(image, model="hog")
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if not face_encodings:
                    logging.warning(f"⚠️ No face found in {face_file.name}")
                    continue
                
                if len(face_encodings) > 1:
                    logging.warning(f"⚠️ Multiple faces found in {face_file.name}, using the largest one")
                    # Use the face with the largest bounding box
                    largest_face_idx = 0
                    largest_area = 0
                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        area = (bottom - top) * (right - left)
                        if area > largest_area:
                            largest_area = area
                            largest_face_idx = i
                    face_encoding = face_encodings[largest_face_idx]
                else:
                    face_encoding = face_encodings[0]
                
                # Get name from filename (remove extension and replace underscores with spaces)
                name = face_file.stem.replace('_', ' ').replace('-', ' ').title()
                
                # Add to known faces
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                
                logging.info(f"✅ Loaded face: {name}")
                
            except Exception as e:
                logging.error(f"❌ Error loading {face_file.name}: {e}")
                continue

        # Cache the encodings if we have any
        if self.known_face_encodings:
            try:
                cache_data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names,
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.face_cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                logging.info(f"💾 Cached {len(self.known_face_names)} face encodings")
            except Exception as e:
                logging.warning(f"Failed to cache face encodings: {e}")

        logging.info(f"🎉 Successfully loaded {len(self.known_face_names)} known faces: {', '.join(self.known_face_names)}")

    def _create_face_instructions(self):
        """Create instructions file for adding faces"""
        known_faces_dir = Path(self.config["paths"]["known_faces"])
        readme_path = known_faces_dir / "README.txt"
        
        instructions = """إضافة وجوه جديدة للنظام - Adding New Faces

العربية:
1. ضع صور الأشخاص هنا بصيغة jpg أو png
2. اسم الملف سيكون اسم الشخص: ahmed_mohamed.jpg
3. استخدم صور واضحة ومواجهة للكاميرا
4. وجه واحد فقط في كل صورة
5. أعد تشغيل البرنامج بعد إضافة الصور

English:
1. Add person photos here in jpg or png format
2. Filename will be the person's name: john_doe.jpg
3. Use clear, front-facing photos
4. One face per image only
5. Restart the program after adding photos

أمثلة - Examples:
- ahmed_mohamed.jpg
- sara_ali.jpg
- john_smith.png
- mary_johnson.jpeg

ملاحظات - Notes:
- يمكن استخدام الشرطة السفلية _ أو الشرطة - بدلاً من المسافات
- سيتم تحويل الاسم تلقائياً: ahmed_mohamed → Ahmed Mohamed
- تأكد من وضوح الصورة وجودة الإضاءة
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logging.info(f"📝 Created instructions: {readme_path}")
    
    def _init_voice_system(self):
        """Initialize voice recognition system"""
        if not self.config["voice"]["enabled"]:
            self.voice_system = None
            return
        
        logging.info("🎤 Initializing voice system...")
        
        try:
            import speech_recognition as sr
            import pyttsx3
            
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.voice_active = False
            self.voice_thread = None
            
            logging.info("✅ Voice system initialized")
            
        except ImportError:
            logging.warning("⚠️ Voice libraries not available, voice features disabled")
            self.voice_system = None
        except Exception as e:
            logging.error(f"❌ Voice system initialization failed: {e}")
            self.voice_system = None
    
    def _init_tracking(self):
        """Initialize person tracking system"""
        logging.info("🔍 Initializing tracking system...")
        
        self.tracked_objects = {}
        self.next_id = 0
        self.max_disappeared = 30
        self.disappeared_counts = {}
        
        logging.info("✅ Tracking system initialized")
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(
                frame, 
                conf=self.config["detection"]["object_confidence"],
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.yolo_model.names[class_id]
                            
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence),
                                'class_name': class_name,
                                'class_id': class_id,
                                'type': 'object'
                            })
            
            return detections
            
        except Exception as e:
            logging.error(f"Object detection error: {e}")
            return []
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect and recognize faces using MediaPipe"""
        if self.face_detection is None:
            return []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(int(bbox.width * w), w - x)
                    height = min(int(bbox.height * h), h - y)
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Extract face ROI for recognition
                    face_roi = frame[y:y+height, x:x+width]
                    name = "Unknown"
                    recognition_confidence = 0.0
                    
                    # Face recognition (improved)
                    if len(self.known_face_encodings) > 0 and face_roi.size > 0:
                        try:
                            # Convert to RGB for face_recognition
                            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                            
                            # Get face encodings with better parameters
                            face_encodings = face_recognition.face_encodings(
                                rgb_face, 
                                known_face_locations=[(0, width, height, 0)],  # Use full ROI
                                num_jitters=1,  # Reduce jitters for speed
                                model="small"   # Use small model for speed
                            )
                            
                            if face_encodings:
                                face_encoding = face_encodings[0]
                                
                                # Compare with known faces
                                face_distances = face_recognition.face_distance(
                                    self.known_face_encodings, face_encoding
                                )
                                
                                # Find best match
                                best_match_index = np.argmin(face_distances)
                                best_distance = face_distances[best_match_index]
                                
                                tolerance = self.config["detection"]["face_recognition_tolerance"]
                                if best_distance <= tolerance:
                                    name = self.known_face_names[best_match_index]
                                    recognition_confidence = max(0.0, 1.0 - best_distance)
                                    
                                    # Log recognition for debugging
                                    logging.debug(f"👤 Recognized: {name} (distance: {best_distance:.3f})")
                                else:
                                    logging.debug(f"❓ Unknown face (best distance: {best_distance:.3f})")
                                    
                        except Exception as e:
                            logging.debug(f"Face recognition error: {e}")
                    
                    faces.append({
                        'bbox': (x, y, x + width, y + height),
                        'name': name,
                        'detection_confidence': detection.score[0],
                        'recognition_confidence': recognition_confidence,
                        'is_known': name != "Unknown",
                        'type': 'face'
                    })
            
            return faces
            
        except Exception as e:
            logging.error(f"Face detection error: {e}")
            return []
    
    def update_tracking(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Update object tracking"""
        if not detections:
            # Mark all objects as disappeared
            for obj_id in list(self.disappeared_counts.keys()):
                self.disappeared_counts[obj_id] += 1
                if self.disappeared_counts[obj_id] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
                    del self.disappeared_counts[obj_id]
            return self.tracked_objects
        
        # Calculate centroids
        input_centroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
        
        # If no existing objects, register all
        if len(self.tracked_objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.tracked_objects[self.next_id] = {
                    'centroid': centroid,
                    'data': detections[i]
                }
                self.disappeared_counts[self.next_id] = 0
                self.next_id += 1
        else:
            # Match existing objects with new detections
            object_centroids = [obj['centroid'] for obj in self.tracked_objects.values()]
            object_ids = list(self.tracked_objects.keys())
            
            if object_centroids and input_centroids:
                # Calculate distance matrix
                D = np.linalg.norm(
                    np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), 
                    axis=2
                )
                
                # Find best matches
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                used_row_indices = set()
                used_col_indices = set()
                
                # Update matched objects
                for (row, col) in zip(rows, cols):
                    if row in used_row_indices or col in used_col_indices:
                        continue
                    
                    if D[row, col] <= 100:  # Distance threshold
                        obj_id = object_ids[row]
                        self.tracked_objects[obj_id] = {
                            'centroid': input_centroids[col],
                            'data': detections[col]
                        }
                        self.disappeared_counts[obj_id] = 0
                        
                        used_row_indices.add(row)
                        used_col_indices.add(col)
                
                # Handle unmatched objects
                unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
                unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
                
                # Mark unmatched existing objects as disappeared
                for row in unused_row_indices:
                    obj_id = object_ids[row]
                    self.disappeared_counts[obj_id] += 1
                    if self.disappeared_counts[obj_id] > self.max_disappeared:
                        del self.tracked_objects[obj_id]
                        del self.disappeared_counts[obj_id]
                
                # Register new objects
                for col in unused_col_indices:
                    self.tracked_objects[self.next_id] = {
                        'centroid': input_centroids[col],
                        'data': detections[col]
                    }
                    self.disappeared_counts[self.next_id] = 0
                    self.next_id += 1
        
        return self.tracked_objects
    
    def draw_detections(self, frame: np.ndarray, objects: List[Dict], 
                       faces: List[Dict], tracked_objects: Dict) -> np.ndarray:
        """Draw all detections and tracking info on frame"""
        display_config = self.config["display"]
        
        # Draw objects
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']
            class_name = obj['class_name']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), display_config["box_thickness"])
            
            # Draw label
            label = f"{class_name}"
            if display_config["show_confidence"]:
                label += f": {confidence:.2f}"
            
            self._draw_label(frame, label, (x1, y1), (0, 255, 0))
        
        # Draw faces
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            name = face['name']
            is_known = face['is_known']
            detection_conf = face['detection_confidence']
            
            # Color based on recognition
            color = (255, 0, 0) if is_known else (0, 0, 255)  # Blue for known, Red for unknown
            
            # Draw face rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, display_config["box_thickness"])
            
            # Draw label
            label = name
            if display_config["show_confidence"]:
                label += f" ({detection_conf:.2f})"
            
            self._draw_label(frame, label, (x1, y1), color)
        
        # Draw tracking info
        for obj_id, obj_data in tracked_objects.items():
            centroid = obj_data['centroid']
            
            # Draw tracking ID
            cv2.putText(frame, f"ID:{obj_id}", 
                       (centroid[0] - 20, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw centroid
            cv2.circle(frame, centroid, 4, (0, 255, 255), -1)
        
        # Draw system info
        if display_config["show_fps"]:
            info_lines = [
                f"FPS: {self.fps:.1f}",
                f"Objects: {len(objects)}",
                f"Faces: {len(faces)}",
                f"Tracked: {len(tracked_objects)}",
                f"Detection: {'ON' if self.detection_active else 'OFF'}"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(frame, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        
        # Draw controls
        controls = "Q:Quit | SPACE:Toggle | S:Screenshot | D:Describe | R:Reload Faces"
        cv2.putText(frame, controls, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_label(self, frame: np.ndarray, label: str, position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw label with background"""
        x, y = position
        font_scale = self.config["display"]["font_scale"]
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        
        # Draw background rectangle
        cv2.rectangle(frame, (x, y - text_height - 10), 
                     (x + text_width + 10, y), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with all AI features"""
        if not self.detection_active:
            return frame
        
        try:
            # Object detection
            objects = self.detect_objects(frame)
            
            # Face detection and recognition
            faces = self.detect_faces(frame)
            
            # Update tracking
            all_detections = objects + faces
            tracked_objects = self.update_tracking(all_detections)
            
            # Draw all detections
            processed_frame = self.draw_detections(frame, objects, faces, tracked_objects)
            
            return processed_frame
            
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return frame
    
    def save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['paths']['screenshots']}/screenshot_{timestamp}.jpg"
            
            cv2.imwrite(filename, frame)
            logging.info(f"📸 Screenshot saved: {filename}")
            
        except Exception as e:
            logging.error(f"Screenshot save error: {e}")
    
    def describe_scene(self, objects: List[Dict], faces: List[Dict]) -> str:
        """Generate scene description"""
        description_parts = []
        
        if objects:
            object_counts = {}
            for obj in objects:
                class_name = obj['class_name']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            object_desc = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                   for name, count in object_counts.items()])
            description_parts.append(f"Objects: {object_desc}")
        
        if faces:
            known_faces = [face['name'] for face in faces if face['is_known']]
            face_desc = f"{len(faces)} face{'s' if len(faces) > 1 else ''}"
            if known_faces:
                face_desc += f" (Known: {', '.join(known_faces)})"
            description_parts.append(f"Faces: {face_desc}")
        
        if not description_parts:
            return "No objects or faces detected"
        
        return " | ".join(description_parts)
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def reload_known_faces(self):
        """Reload known faces from directory"""
        logging.info("🔄 Reloading known faces...")
        
        # Delete cache to force reload
        if self.face_cache_file.exists():
            try:
                self.face_cache_file.unlink()
                logging.info("🗑️ Cleared face cache")
            except Exception as e:
                logging.warning(f"Failed to clear cache: {e}")
        
        # Reload faces
        self._load_known_faces()
        
        logging.info(f"✅ Reloaded {len(self.known_face_names)} faces")
    
    def run(self):
        """Main system loop"""
        if self.camera is None:
            logging.error("❌ Cannot start - camera not available")
            return
        
        logging.info("🚀 Starting Smart Vision System...")
        logging.info("Controls: Q=Quit, SPACE=Toggle Detection, S=Screenshot")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Smart Vision System', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.detection_active = not self.detection_active
                    status = "ON" if self.detection_active else "OFF"
                    logging.info(f"Detection {status}")
                elif key == ord('s'):
                    self.save_screenshot(processed_frame)
                elif key == ord('d'):
                    # Describe current scene
                    objects = self.detect_objects(frame)
                    faces = self.detect_faces(frame)
                    description = self.describe_scene(objects, faces)
                    logging.info(f"Scene: {description}")
                elif key == ord('r'):
                    self.reload_known_faces()
        
        except KeyboardInterrupt:
            logging.info("Stopped by user")
        except Exception as e:
            logging.error(f"System error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system"""
        logging.info("🛑 Stopping Smart Vision System...")
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        logging.info("✅ Smart Vision System stopped")

def main():
    """Main function"""
    try:
        system = SmartVisionSystem()
        system.run()
    except Exception as e:
        logging.error(f"Failed to start system: {e}")

if __name__ == "__main__":
    main()
