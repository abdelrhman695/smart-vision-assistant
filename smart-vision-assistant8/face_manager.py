"""
Face Management GUI - إدارة الوجوه بواجهة رسومية
Simple GUI for managing known faces
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import threading
from pathlib import Path
import face_recognition
from PIL import Image, ImageTk
import numpy as np

class FaceManagerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Manager - مدير الوجوه")
        self.root.geometry("800x600")
        
        self.known_faces_dir = "known_faces"
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        self.camera = None
        self.camera_active = False
        
        self.setup_ui()
        self.refresh_face_list()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Manager - مدير الوجوه", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Face list
        left_frame = ttk.LabelFrame(main_frame, text="Known Faces - الوجوه المعروفة", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Face listbox
        self.face_listbox = tk.Listbox(left_frame, height=15)
        self.face_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.face_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.face_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Buttons for face management
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Delete Selected\nحذف المحدد", 
                  command=self.delete_selected_face).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Refresh\nتحديث", 
                  command=self.refresh_face_list).grid(row=0, column=1, padx=5)
        
        # Right panel - Add new face
        right_frame = ttk.LabelFrame(main_frame, text="Add New Face - إضافة وجه جديد", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Name entry
        ttk.Label(right_frame, text="Name - الاسم:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.name_entry = ttk.Entry(right_frame, width=30)
        self.name_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Add from file
        ttk.Button(right_frame, text="Add from File\nإضافة من ملف", 
                  command=self.add_from_file).grid(row=2, column=0, columnspan=2, 
                                                  sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Camera section
        ttk.Separator(right_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, 
                                                           sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(right_frame, text="Camera Capture - التقاط من الكاميرا", 
                 font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        # Camera preview
        self.camera_label = ttk.Label(right_frame, text="Camera Preview\nمعاينة الكاميرا", 
                                     background="gray", width=40)
        self.camera_label.grid(row=5, column=0, columnspan=2, pady=(0, 10))
        
        # Camera buttons
        camera_button_frame = ttk.Frame(right_frame)
        camera_button_frame.grid(row=6, column=0, columnspan=2)
        
        self.start_camera_btn = ttk.Button(camera_button_frame, text="Start Camera\nتشغيل الكاميرا", 
                                          command=self.start_camera)
        self.start_camera_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.capture_btn = ttk.Button(camera_button_frame, text="Capture\nالتقاط", 
                                     command=self.capture_face, state="disabled")
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        self.stop_camera_btn = ttk.Button(camera_button_frame, text="Stop Camera\nإيقاف الكاميرا", 
                                         command=self.stop_camera, state="disabled")
        self.stop_camera_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - جاهز")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
    
    def refresh_face_list(self):
        """Refresh the list of known faces"""
        self.face_listbox.delete(0, tk.END)
        
        face_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            face_files.extend(Path(self.known_faces_dir).glob(ext))
        
        for face_file in sorted(face_files):
            name = face_file.stem.replace('_', ' ').title()
            self.face_listbox.insert(tk.END, f"{name} ({face_file.name})")
        
        self.status_var.set(f"Found {len(face_files)} known faces - تم العثور على {len(face_files)} وجه معروف")
    
    def delete_selected_face(self):
        """Delete the selected face"""
        selection = self.face_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning - تحذير", "Please select a face to delete\nيرجى اختيار وجه للحذف")
            return
        
        selected_text = self.face_listbox.get(selection[0])
        filename = selected_text.split('(')[1].split(')')[0]
        filepath = Path(self.known_faces_dir) / filename
        
        if messagebox.askyesno("Confirm Delete - تأكيد الحذف", 
                              f"Delete {filename}?\nحذف {filename}؟"):
            try:
                filepath.unlink()
                self.refresh_face_list()
                self.status_var.set(f"Deleted {filename} - تم حذف {filename}")
            except Exception as e:
                messagebox.showerror("Error - خطأ", f"Failed to delete file: {e}")
    
    def add_from_file(self):
        """Add a face from an image file"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning - تحذير", "Please enter a name\nيرجى إدخال اسم")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image - اختر صورة",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Verify face in image
            image = face_recognition.load_image_file(file_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                messagebox.showerror("Error - خطأ", "No face found in the image\nلم يتم العثور على وجه في الصورة")
                return
            
            if len(face_locations) > 1:
                if not messagebox.askyesno("Multiple Faces - عدة وجوه", 
                                          "Multiple faces found. Continue with the first one?\nتم العثور على عدة وجوه. المتابعة مع الأول؟"):
                    return
            
            # Save the image
            filename = f"{name.replace(' ', '_').lower()}.jpg"
            output_path = Path(self.known_faces_dir) / filename
            
            # Load and save with OpenCV
            cv_image = cv2.imread(file_path)
            cv2.imwrite(str(output_path), cv_image)
            
            self.refresh_face_list()
            self.name_entry.delete(0, tk.END)
            self.status_var.set(f"Added {name} - تم إضافة {name}")
            
            messagebox.showinfo("Success - نجح", f"Face added successfully!\nتم إضافة الوجه بنجاح!\n\nRestart the vision system to use the new face.\nأعد تشغيل نظام الرؤية لاستخدام الوجه الجديد.")
            
        except Exception as e:
            messagebox.showerror("Error - خطأ", f"Failed to add face: {e}")
    
    def start_camera(self):
        """Start camera preview"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error - خطأ", "Cannot open camera\nلا يمكن فتح الكاميرا")
                return
            
            self.camera_active = True
            self.start_camera_btn.config(state="disabled")
            self.capture_btn.config(state="normal")
            self.stop_camera_btn.config(state="normal")
            
            self.update_camera_preview()
            self.status_var.set("Camera started - تم تشغيل الكاميرا")
            
        except Exception as e:
            messagebox.showerror("Error - خطأ", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera preview"""
        self.camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_camera_btn.config(state="normal")
        self.capture_btn.config(state="disabled")
        self.stop_camera_btn.config(state="disabled")
        
        # Reset camera preview
        self.camera_label.config(image="", text="Camera Preview\nمعاينة الكاميرا")
        self.status_var.set("Camera stopped - تم إيقاف الكاميرا")
    
    def update_camera_preview(self):
        """Update camera preview"""
        if not self.camera_active or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Detect faces for preview
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize for preview
            frame_pil = frame_pil.resize((320, 240), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(frame_pil)
            
            # Update label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep a reference
        
        # Schedule next update
        if self.camera_active:
            self.root.after(30, self.update_camera_preview)
    
    def capture_face(self):
        """Capture face from camera"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning - تحذير", "Please enter a name\nيرجى إدخال اسم")
            return
        
        if not self.camera or not self.camera_active:
            messagebox.showerror("Error - خطأ", "Camera not active\nالكاميرا غير نشطة")
            return
        
        ret, frame = self.camera.read()
        if not ret:
            messagebox.showerror("Error - خطأ", "Failed to capture frame\nفشل في التقاط الإطار")
            return
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Check for faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            messagebox.showwarning("Warning - تحذير", "No face detected!\nلم يتم اكتشاف وجه!")
            return
        elif len(faces) > 1:
            messagebox.showwarning("Warning - تحذير", "Multiple faces detected! Please ensure only one face is visible.\nتم اكتشاف عدة وجوه! تأكد من ظهور وجه واحد فقط.")
            return
        
        try:
            # Save the frame
            filename = f"{name.replace(' ', '_').lower()}.jpg"
            output_path = Path(self.known_faces_dir) / filename
            cv2.imwrite(str(output_path), frame)
            
            # Verify with face_recognition
            image = face_recognition.load_image_file(str(output_path))
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                output_path.unlink()  # Delete the file
                messagebox.showerror("Error - خطأ", "Could not encode face, please try again\nلا يمكن ترميز الوجه، حاول مرة أخرى")
                return
            
            self.refresh_face_list()
            self.name_entry.delete(0, tk.END)
            self.status_var.set(f"Captured {name} - تم التقاط {name}")
            
            messagebox.showinfo("Success - نجح", f"Face captured successfully!\nتم التقاط الوجه بنجاح!\n\nRestart the vision system to use the new face.\nأعد تشغيل نظام الرؤية لاستخدام الوجه الجديد.")
            
        except Exception as e:
            messagebox.showerror("Error - خطأ", f"Failed to capture face: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.camera_active:
            self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceManagerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
