"""
YOLO Text Detection + OCR System - LABVIEW REAL-TIME INTEGRATION
Complete real-time integration: Initialize once, process continuously
Returns detection results in real-time (synchronized)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import os
import sys
import time
import logging
from collections import deque

# ==================== Suppress YOLO Logs ====================
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# ==================== FPS COUNTER ====================
class FPSCounter:
    """Efficient FPS counter with rolling average"""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

# ==================== Tesseract Configuration ====================
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + ';' + os.environ.get('PATH', '')

# ==================== GLOBAL SESSION STATE ====================
_session = {
    'camera': None,
    'model': None,
    'initialized': False,
    'running': False,
    'fps_counter': FPSCounter(),
    'all_detections': [],
    'window_positions': {}
}

# ==================== HELPER FUNCTIONS ====================

def preprocess_image_for_ocr(image, enhancement_level=2):
    """Preprocess image for OCR"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape[:2]
        if h < 30:
            scale = 30 / h
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), 
                             interpolation=cv2.INTER_CUBIC)
        
        if enhancement_level == 1:
            return gray
        elif enhancement_level == 2:
            return cv2.bilateralFilter(gray, 5, 50, 50)
        else:
            denoised = cv2.bilateralFilter(gray, 5, 50, 50)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(20, 20))
            return clahe.apply(denoised)
    except Exception as e:
        return gray

def extract_text_ocr(image, psm=6, enhancement=2):
    """Extract text with Tesseract"""
    try:
        preprocessed = preprocess_image_for_ocr(image, enhancement_level=max(enhancement, 2))
        
        if preprocessed is None:
            return ""
        
        config = f'--psm {psm} --oem 1'
        text = pytesseract.image_to_string(preprocessed, config=config, lang='eng')
        text = ' '.join(text.split())
        
        return text.strip()
    except Exception as e:
        return ""

def crop_box_from_frame(frame, box):
    """Crop region from frame"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    
    cropped = frame[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)

def display_detection_window(cropped_image, text, zone_id):
    """Display detection window with text"""
    global _session
    
    try:
        display = cropped_image.copy()
        h, w = display.shape[:2]
        
        # Calculate text height
        if text and text.strip():
            lines = []
            remaining = text.strip()
            while len(remaining) > 60:
                lines.append(remaining[:60])
                remaining = remaining[60:]
            if remaining:
                lines.append(remaining)
            
            text_height = max(80, 30 + len(lines) * 25)
        else:
            text_height = 80
            lines = []
        
        # Add text border
        display = cv2.copyMakeBorder(display, text_height, 10, 10, 10, 
                                    cv2.BORDER_CONSTANT, value=(220, 220, 220))
        
        # Display text
        y_offset = 30
        if text and text.strip():
            for i, line in enumerate(lines):
                try:
                    cv2.putText(display, line, (15, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                except:
                    line_safe = line.encode('ascii', 'ignore').decode('ascii')
                    cv2.putText(display, line_safe, (15, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(display, "NO TEXT DETECTED", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 0), 1)
        
        cv2.putText(display, f"Zone {zone_id}", (15, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 1)
        
        window_name = f"Zone {zone_id}"
        cv2.imshow(window_name, display)
        
        # Position window
        if window_name not in _session['window_positions']:
            x_pos = 50 + (zone_id % 2) * 700
            y_pos = 50 + (zone_id // 2) * 450
            cv2.moveWindow(window_name, x_pos, y_pos)
            _session['window_positions'][window_name] = (x_pos, y_pos)
        
    except Exception as e:
        pass

# ==================== LABVIEW API FUNCTIONS ====================

def initialize(camera_number, resolution_camera, ai_model_path):
    """
    INITIALIZE ONCE - Setup camera and model
    
    Parameters:
    -----------
    camera_number : int
        Camera ID (0, 1, 2, etc.)
    resolution_camera : str
        Resolution as "width,height" (e.g., "1280,720")
    ai_model_path : str
        Path to YOLO model file
    
    Returns:
    --------
    tuple: (success, error_message)
        - (True, "") if initialization successful
        - (False, "error message") if failed
    """
    global _session
    
    try:
        # Parse resolution
        parts = resolution_camera.split(',')
        width = int(parts[0])
        height = int(parts[1])
        
        # Validate model
        if not os.path.exists(ai_model_path):
            return (False, f"Model not found: {ai_model_path}")
        
        # Load YOLO model
        _session['model'] = YOLO(ai_model_path)
        
        # Initialize camera
        _session['camera'] = cv2.VideoCapture(camera_number)
        if not _session['camera'].isOpened():
            return (False, f"Cannot open camera {camera_number}")
        
        # Set resolution
        if width > 0 and height > 0:
            _session['camera'].set(cv2.CAP_PROP_FRAME_WIDTH, width)
            _session['camera'].set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        _session['camera'].set(cv2.CAP_PROP_FPS, 30)
        
        # Reset state
        _session['initialized'] = True
        _session['running'] = True
        _session['fps_counter'] = FPSCounter()
        _session['all_detections'] = []
        _session['window_positions'] = {}
        
        return (True, "")
        
    except Exception as e:
        _session['initialized'] = False
        return (False, str(e))


def process_frame():
    """
    CONTINUOUS PROCESSING - Call in loop
    Get one frame, run detection, return result in real-time
    
    Returns:
    --------
    str: Detection result
        - Detected text (if OCR found text)
        - "not detected" (if no text found)
        - "Error: message" (if error occurred)
        - "CAMERA_CLOSED" (if camera is closed/not initialized)
    """
    global _session
    
    try:
        # Check if initialized
        if not _session['initialized'] or _session['camera'] is None:
            return str("Error: Not initialized. Call initialize() first")
        
        # Read frame
        success, frame = _session['camera'].read()
        if not success:
            return str("Error: Cannot read frame from camera")
        
        _session['fps_counter'].update()
        
        # YOLO Detection (silent mode) - non-persistent: only current-frame detections
        results = _session['model'].track(frame, classes=0, conf=0.6, imgsz=640, 
                                          persist=False, verbose=False)
        num_detections = len(results[0].boxes)
        
        # Annotated frame
        annotated_frame = results[0].plot()
        
        # Process detections
        detected_texts = []
        
        if num_detections > 0:
            for idx, box in enumerate(results[0].boxes):
                cropped, coords = crop_box_from_frame(frame, box)
                
                if cropped.size > 0:
                    # OCR Extraction
                    text = extract_text_ocr(cropped, psm=6, enhancement=2)
                    
                    # Display detection window
                    display_detection_window(cropped, text, idx + 1)
                    
                    # Store detection if text found
                    if text.strip():
                        detected_texts.append(text)
                        _session['all_detections'].append(text)

        # Cleanup stale Zone windows: remove any previously-created Zone windows
        # that are no longer present in the current frame so IDs don't persist.
        try:
            current_zone_names = set(f"Zone {i+1}" for i in range(num_detections))
            existing_windows = set(_session['window_positions'].keys())
            to_remove = existing_windows - current_zone_names
            for window in to_remove:
                try:
                    cv2.destroyWindow(window)
                except Exception:
                    pass
                _session['window_positions'].pop(window, None)
        except Exception:
            pass

        # Display main stream
        current_fps = _session['fps_counter'].get_fps()
        cv2.putText(annotated_frame, f"Zones: {num_detections}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Press 'q' to stop, 'c' to clear windows", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("YOLO Detection Live", annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _session['running'] = False
            return str("STOPPED_BY_USER")
        elif key == ord('c'):
            for window in list(_session['window_positions'].keys()):
                cv2.destroyWindow(window)
            _session['window_positions'].clear()
        
        # Return result
        if detected_texts:
            return str("|".join(detected_texts))  # Multiple detections separated by |
        else:
            return str("not detected")
        
    except Exception as e:
        return str(f"Error: {str(e)}")


def cleanup():
    """
    CLEANUP - Call when done
    Close camera and release resources
    
    Returns:
    --------
    bool: True if cleanup successful
    """
    global _session
    
    try:
        _session['running'] = False
        
        if _session['camera']:
            _session['camera'].release()
        
        cv2.destroyAllWindows()
        
        # Print summary
        if _session['all_detections']:
            print("\n" + "="*60)
            print("DETECTIONS SUMMARY:")
            for i, text in enumerate(_session['all_detections'], 1):
                print(f"{i}. {text}")
            print(f"\nFinal FPS: {_session['fps_counter'].get_fps():.1f}")
            print("="*60)
        
        _session['initialized'] = False
        return True
        
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return False


def get_session_info():
    """
    GET SESSION INFO - For debugging
    
    Returns:
    --------
    tuple: (is_initialized, is_running, fps, total_detections)
    """
    global _session
    
    return (
        _session['initialized'],
        _session['running'],
        _session['fps_counter'].get_fps(),
        len(_session['all_detections'])
    )


# ==================== BACKWARD COMPATIBILITY ====================

def main():
    """Legacy main function for command-line testing"""
    print("LabVIEW Real-time Integration Module")
    print("Use initialize(), process_frame(), and cleanup() functions")
    print("\nExample usage:")
    print("  1. Call: initialize(0, '1280,720', 'my_model.pt')")
    print("  2. Loop: result = process_frame()")
    print("  3. Call: cleanup()")

if __name__ == '__main__':
    main()
