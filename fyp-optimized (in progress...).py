import time
import os
import threading
import queue
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import cv2  # For optional display and drawing overlays
from picamera2 import Picamera2
from ultralytics import YOLO
from gtts import gTTS
from PIL import Image
import pygame

# --------------------- CONFIGURATION ---------------------
@dataclass
class AppConfig:
    # Camera configuration
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 15
    camera_fov: float = 62.2  # Degrees (Raspberry Pi V2 camera)
    
    # Detection parameters
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    focal_length: float = 3.04  # mm (Raspberry Pi V2 camera)
    sensor_width: float = 3.68  # mm
    known_heights: Dict[int, float] = field(default_factory=lambda: {
        0: 1.7,   # person
        2: 1.5,   # car
        3: 2.5,   # motorcycle
        5: 2.0,   # bus
        7: 0.5    # truck (lower estimate)
    })
    
    # Alert thresholds (meters)
    critical_distance: float = 1.5
    caution_distance: float = 3.0
    alert_cooldown: int = 8  # seconds
    
    # System parameters
    max_queue_size: int = 3
    frame_processing_interval: float = 0.1
    audio_priority_override: bool = True
    
    # Optional: specify a dedicated beep file (or leave blank to use TTS for beep)
    beep_audio_file: str = "beep.mp3"
    
    # Development display configuration (only used if a monitor is attached)
    display_feed: bool = False  # Set to True ONLY during development/when a display is available
    display_window_name: str = "Live Feed"
    
    # Audio feedback when path is clear
    path_clearance_interval: int = 10  # seconds

config = AppConfig()

# --------------------- AUDIO UTILITIES ---------------------
class AudioCache:
    """Cache for frequently used audio phrases."""
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.cache = {}
            cls._instance.cache_dir = "./audio_cache"
            os.makedirs(cls._instance.cache_dir, exist_ok=True)
        return cls._instance
    
    def get_audio(self, text: str) -> str:
        """Return the path to a cached audio file or generate it if missing."""
        safe_name = "".join(c if c.isalnum() else "_" for c in text)
        path = os.path.join(self.cache_dir, f"{safe_name}.mp3")
        
        if not os.path.exists(path):
            # For empty text (used for beep), try using the dedicated beep file if available.
            if text.strip() == "":
                if os.path.exists(config.beep_audio_file):
                    return config.beep_audio_file
                else:
                    tts = gTTS("beep", lang='en')
                    tts.save(path)
            else:
                tts = gTTS(text, lang='en')
                tts.save(path)
        return path

class PriorityAudioQueue:
    """Queue system with priority handling for audio messages."""
    def __init__(self):
        self._critical_queue = queue.Queue(maxsize=2)
        self._normal_queue = queue.Queue(maxsize=5)
        
    def put(self, item: str, priority: bool = False):
        if priority:
            if self._critical_queue.full():
                try:
                    self._critical_queue.get_nowait()
                except queue.Empty:
                    pass
            self._critical_queue.put_nowait(item)
        else:
            if self._normal_queue.full():
                try:
                    self._normal_queue.get_nowait()
                except queue.Empty:
                    pass
            self._normal_queue.put_nowait(item)
            
    def get(self, timeout: float = None) -> str:
        try:
            return self._critical_queue.get_nowait()
        except queue.Empty:
            return self._normal_queue.get(timeout=timeout)

# --------------------- CAMERA COMPONENT ---------------------
class CameraController:
    """Managed camera interface with hardware optimizations."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.picam2 = Picamera2()
        self._configure_camera()
        self.frame_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._stop_event = threading.Event()

    def _configure_camera(self):
        camera_config = self.picam2.create_preview_configuration(
            main={
                "format": "RGB888",
                "size": self.config.camera_resolution
            },
            controls={"FrameRate": self.config.camera_fps}
        )
        self.picam2.configure(camera_config)
        
    def start(self):
        self.picam2.start()
        logging.info("Camera started, warming up...")
        time.sleep(2)  # Allow time for camera warm-up
        threading.Thread(target=self._capture_loop, daemon=True).start()
        
    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self.picam2.capture_array("main")
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            except Exception as e:
                logging.error(f"Camera capture error: {e}")
                time.sleep(1)
                
    def stop(self):
        self._stop_event.set()
        self.picam2.stop()
        logging.info("Camera stopped.")

# --------------------- OBJECT DETECTION ---------------------
class ObjectDetector:
    """Enhanced object detection with distance estimation."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = YOLO(self.config.model_path)
        self.tracker: Dict[str, Any] = {}  # object_id: (last_seen, distance)
        self.px_per_mm = self.config.camera_resolution[0] / self.config.sensor_width
        
    def _calculate_distance(self, cls_id: int, bbox_height: float) -> Optional[float]:
        """Estimate distance using perspective projection."""
        if cls_id not in self.config.known_heights:
            return None
            
        object_height = self.config.known_heights[cls_id]
        sensor_height_mm = (self.config.sensor_width / self.config.camera_resolution[0]) * self.config.camera_resolution[1]
        image_height_mm = bbox_height / self.px_per_mm
        if image_height_mm == 0:
            return None
        distance_m = (object_height * self.config.focal_length) / (image_height_mm * 1e-3)
        return distance_m
        
    def process_frame(self, frame: np.ndarray):
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            if box.conf < self.config.confidence_threshold:
                continue
                
            cls_id = int(box.cls)
            bbox = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            bbox_height = bbox[3] - bbox[1]
            
            distance = self._calculate_distance(cls_id, bbox_height)
            if distance is None:
                continue
            
            detections.append({
                "cls_id": cls_id,
                "distance": distance,
                "position": (x_center, y_center),
                "timestamp": time.time(),
                "bbox": bbox
            })
            
        return detections

# --------------------- AUDIO COMPONENT ---------------------
class AudioController:
    """Optimized audio playback system."""
    def __init__(self, config: AppConfig):
        self.config = config
        pygame.mixer.init()
        self.cache = AudioCache()
        
    def play_audio(self, text: str, priority: bool = False):
        """Play audio with optional priority override."""
        if self.config.audio_priority_override and priority:
            pygame.mixer.music.stop()
            
        file_path = self.cache.get_audio(text)
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            logging.error(f"Audio playback error: {e}")

# --------------------- MAIN APPLICATION ---------------------
class AssistiveVisionSystem:
    """Main application for the assistive vision system."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.camera = CameraController(config)
        self.detector = ObjectDetector(config)
        self.audio = AudioController(config)
        self.audio_queue = PriorityAudioQueue()
        self._stop_event = threading.Event()
        self.last_path_clearance_audio = time.time()
        
    def is_in_center(self, detection):
        """Return True if the detection is roughly in the center of the frame."""
        x_center = detection['position'][0]
        frame_center = self.config.camera_resolution[0] / 2
        tolerance = 0.2 * frame_center  # ±20% of half the frame width
        return abs(x_center - frame_center) < tolerance
        
    def _process_detections(self, detections):
        current_time = time.time()
        center_alert_triggered = False
        
        for obj in detections:
            if not self.is_in_center(obj):
                continue  # Only process objects roughly in the center
            
            obj_id = self._generate_object_id(obj)
            last_seen, _ = self.detector.tracker.get(obj_id, (0, None))
            self.detector.tracker[obj_id] = (current_time, obj['distance'])
            
            if obj['distance'] <= self.config.critical_distance:
                self.audio_queue.put("BEEP", priority=True)
                center_alert_triggered = True
            elif obj['distance'] <= self.config.caution_distance:
                if (current_time - last_seen) > self.config.alert_cooldown:
                    object_name = self.detector.model.names.get(obj['cls_id'], "object")
                    self.audio_queue.put(f"Caution: {object_name} ahead", priority=False)
                    center_alert_triggered = True
                    
        return center_alert_triggered
        
    def _generate_object_id(self, obj):
        """Generate a unique ID based on object class and rough position."""
        return f"{obj['cls_id']}-{int(obj['position'][0] / 50)}-{int(obj['position'][1] / 50)}"
        
    def _audio_handler(self):
        while not self._stop_event.is_set():
            try:
                msg = self.audio_queue.get(timeout=1)
                if msg == "BEEP":
                    self.audio.play_audio("", priority=True)
                else:
                    self.audio.play_audio(msg, priority=False)
            except queue.Empty:
                continue
                
    def _cleanup_tracker(self):
        """Remove stale tracked objects to avoid redundant alerts."""
        current_time = time.time()
        stale_keys = [k for k, (t, _) in self.detector.tracker.items()
                      if current_time - t > self.config.alert_cooldown * 2]
        for k in stale_keys:
            del self.detector.tracker[k]
            
    def display_frame(self, frame: np.ndarray, detections: list):
        """Overlay detection information on the frame and display it (optional)."""
        frame_copy = frame.copy()
        for detection in detections:
            bbox = detection["bbox"].astype(int)
            x_min, y_min, x_max, y_max = bbox
            # Color coding based on alert level
            if detection["distance"] <= self.config.critical_distance:
                color = (0, 0, 255)  # Red
            elif detection["distance"] <= self.config.caution_distance:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), color, 2)
            object_name = self.detector.model.names.get(detection["cls_id"], "object")
            text = f"{object_name}: {detection['distance']:.1f}m"
            cv2.putText(frame_copy, text, (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.config.display_window_name, frame_bgr)
        cv2.waitKey(1)
        
    def run(self):
        logging.info("Starting Assistive Vision System...")
        self.camera.start()
        threading.Thread(target=self._audio_handler, daemon=True).start()
        
        # Boot-up audio messages
        self.audio_queue.put("Welcome user. Please wait for the program to start.", priority=False)
        time.sleep(2)
        self.audio_queue.put("Program has started. Kindly wait for path clearance.", priority=False)
        self.last_path_clearance_audio = time.time()
        
        # Create display window only if enabled
        if self.config.display_feed:
            cv2.namedWindow(self.config.display_window_name, cv2.WINDOW_NORMAL)
        
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.camera.frame_queue.get(timeout=1)
                    detections = self.detector.process_frame(frame)
                    center_alert = self._process_detections(detections)
                    
                    # If no center alert for a period, give a "path clearance" audio message
                    if not center_alert and (time.time() - self.last_path_clearance_audio > self.config.path_clearance_interval):
                        self.audio_queue.put("Kindly wait for path clearance.", priority=False)
                        self.last_path_clearance_audio = time.time()
                        
                    self._cleanup_tracker()
                    
                    if self.config.display_feed:
                        self.display_frame(frame, detections)
                        
                    time.sleep(self.config.frame_processing_interval)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Main processing error: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Shutting down...")
            self.shutdown()
            
    def shutdown(self):
        self._stop_event.set()
        self.camera.stop()
        pygame.mixer.quit()
        if self.config.display_feed:
            cv2.destroyAllWindows()
        logging.info("System shutdown complete.")

# --------------------- ENTRY POINT ---------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("system.log"),
            logging.StreamHandler()
        ]
    )
    
    avs = AssistiveVisionSystem(config)
    avs.run()

if __name__ == '__main__':
    main()

