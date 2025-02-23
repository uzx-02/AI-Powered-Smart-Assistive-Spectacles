import time
import os
import threading
import queue
import logging
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from gtts import gTTS
from PIL import Image

# ------------- CONFIGURATION -------------
# Distance thresholds (bounding box area)
NORMAL_DISTANCE_THRESHOLD = 5000
CRITICAL_DISTANCE_THRESHOLD = 15000
ALERT_DELAY = 10  # seconds delay for caution messages

# Debug mode: if True, frames will be saved (for debugging) in batches of 5 then deleted.
DEBUG_MODE = False  
FRAME_SAVE_COUNT = 5  
FRAME_SAVE_DIR = "./debug_frames"
if DEBUG_MODE and not os.path.exists(FRAME_SAVE_DIR):
    os.makedirs(FRAME_SAVE_DIR)

# ------------- LOGGING SETUP -------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# ------------- GLOBAL VARIABLES & QUEUES -------------
frame_queue = queue.Queue(maxsize=5)  # to hold captured frames
audio_queue = queue.Queue()           # to hold audio messages
detected_objects = {}  # object_key -> last caution time

# Event to signal threads to stop gracefully.
stop_event = threading.Event()

# ------------- INITIALIZE YOLO MODEL -------------
model = YOLO("yolov8n.pt")  # Ensure the model is downloaded

# ------------- AUDIO FEEDBACK FUNCTIONS -------------
def give_audio_feedback(message):
    try:
        tts = gTTS(message, lang='en')
        feedback_file = "feedback.mp3"
        tts.save(feedback_file)
        os.system(f"mpg123 {feedback_file}")
    except Exception as e:
        logging.error(f"Error in audio feedback: {e}")

def play_beep():
    try:
        os.system("mpg123 beep.mp3")
    except Exception as e:
        logging.error(f"Error playing beep: {e}")

# ------------- AUDIO THREAD -------------
def audio_thread_func():
    while not stop_event.is_set():
        try:
            msg = audio_queue.get(timeout=1)
            if msg == "BEEP":
                play_beep()
            else:
                give_audio_feedback(msg)
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Audio thread error: {e}")

# ------------- CAMERA CAPTURE THREAD -------------
def camera_thread_func():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        logging.info("Camera initialized successfully.")
        time.sleep(2)  # Allow camera to stabilize
        
        while not stop_event.is_set():
            frame = picam2.capture_array()
            # Ensure frame is in RGB format (drop alpha if present)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            # Place the frame in the queue (drop oldest if necessary)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
    except Exception as e:
        logging.error(f"Camera thread error: {e}")
    finally:
        try:
            picam2.stop()
        except Exception as e:
            logging.error(f"Error stopping camera: {e}")

# ------------- HELPER FUNCTION -------------
def is_in_front(bbox_center_x, frame_width):
    """Return True if the detected object is near the center of the frame."""
    center_tolerance = 0.2  # ±20%
    frame_center = frame_width / 2
    return (frame_center * (1 - center_tolerance) <= bbox_center_x <= frame_center * (1 + center_tolerance))

# ------------- DETECTION THREAD -------------
def detection_thread_func():
    saved_frame_count = 0
    saved_frame_paths = []
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            results = model(frame)
        except Exception as e:
            logging.error(f"Detection error: {e}")
            continue

        for result in results:
            for box in result.boxes:
                try:
                    cls_id = int(box.cls[0])
                    # Confidence could be used to filter out low-confidence detections if needed.
                    conf = box.conf[0]
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    box_area = (x_max - x_min) * (y_max - y_min)
                    bbox_center_x = (x_min + x_max) / 2
                    frame_width = frame.shape[1]
                    
                    # Process only objects in the center path.
                    if not is_in_front(bbox_center_x, frame_width):
                        continue

                    object_key = f"{cls_id}-{x_min}-{y_min}-{x_max}-{y_max}"
                    current_time = time.time()

                    if box_area >= CRITICAL_DISTANCE_THRESHOLD:
                        # If object is too close, immediately alert with beep.
                        audio_queue.put("BEEP")
                        # Remove any previous caution alert for this object.
                        detected_objects.pop(object_key, None)
                    elif box_area >= NORMAL_DISTANCE_THRESHOLD:
                        # For normal alert, check if we have already alerted recently.
                        if object_key in detected_objects and (current_time - detected_objects[object_key] < ALERT_DELAY):
                            continue
                        else:
                            detected_objects[object_key] = current_time
                            # Retrieve object name; if not available, fallback to generic name.
                            object_name = model.names.get(cls_id, "object")
                            audio_queue.put(f"Caution. {object_name} detected ahead.")
                            
                            # Optionally save frame for debugging.
                            if DEBUG_MODE:
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                filename = os.path.join(FRAME_SAVE_DIR, f"frame_{timestamp}.jpg")
                                im = Image.fromarray(frame)
                                im.save(filename)
                                saved_frame_paths.append(filename)
                                saved_frame_count += 1
                                
                                # Once 5 frames are saved, delete them.
                                if saved_frame_count >= FRAME_SAVE_COUNT:
                                    for path in saved_frame_paths:
                                        try:
                                            os.remove(path)
                                        except Exception as e:
                                            logging.error(f"Error deleting frame {path}: {e}")
                                    saved_frame_paths = []
                                    saved_frame_count = 0
                except Exception as e:
                    logging.error(f"Error processing detection box: {e}")

        # Slight pause to reduce CPU load
        time.sleep(0.1)

# ------------- MAIN EXECUTION -------------
if __name__ == '__main__':
    logging.info("Starting system threads...")

    camera_thread = threading.Thread(target=camera_thread_func, daemon=True)
    detection_thread = threading.Thread(target=detection_thread_func, daemon=True)
    audio_thread = threading.Thread(target=audio_thread_func, daemon=True)

    camera_thread.start()
    detection_thread.start()
    audio_thread.start()

    # Boot-up audio feedback
    try:
        audio_queue.put("Welcome user. Please wait for the program to start.")
        time.sleep(2)
        audio_queue.put("Program has started. Kindly wait for path clearance.")
    except Exception as e:
        logging.error(f"Error in boot-up audio: {e}")

    # Main loop simply waits until keyboard interrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Stopping threads...")
        stop_event.set()
        camera_thread.join()
        detection_thread.join()
        audio_thread.join()
        logging.info("System shutdown complete.")
