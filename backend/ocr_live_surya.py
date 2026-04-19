import cv2
import urllib.request
import numpy as np
import threading
import queue
import socket
import time
import ssl
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# Bypass SSL certificate verification for macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Surya OCR (better accuracy than EasyOCR)
print("Loading Surya OCR models...")
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()
print("Models loaded!\n")

# Camera configuration
base_url = "http://10.139.247.40:8080"
endpoints = ["/video", "/mjpegfeed", "/stream", "/axis-cgi/mjpg/video.cgi", ""]

url = None
for endpoint in endpoints:
    test_url = base_url + endpoint
    try:
        req = urllib.request.Request(test_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        stream = urllib.request.urlopen(req, timeout=3)
        content_type = stream.headers.get('Content-Type', '')
        if 'mjpeg' in content_type.lower() or 'application/octet-stream' in content_type.lower():
            url = test_url
            stream.close()
            break
        stream.close()
    except Exception:
        pass

if not url:
    url = base_url + "/video"

print(f"Using camera: {url}\n")
print("Starting live OCR stream...")
print("Press ESC to exit\n")

# Threading setup
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def read_stream():
    """Background thread: reads MJPEG stream"""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Connection': 'close'
        })
        stream = urllib.request.urlopen(req, timeout=10)
        
        try:
            sock = stream.fp._sock
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (AttributeError, OSError):
            pass
        
        bytes_data = b''
        
        while not stop_event.is_set():
            chunk = stream.read(32768)
            if not chunk:
                break
            
            bytes_data += chunk
            
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9', a)
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    try:
                        frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
        
        stream.close()
    except Exception as e:
        print(f"Stream error: {e}")

# Start camera thread
thread = threading.Thread(target=read_stream, daemon=True)
thread.start()

try:
    fps_timer = time.time()
    frame_count = 0
    
    # Store OCR results
    ocr_results = []
    process_lock = threading.Lock()
    is_processing = False
    
    def ocr_thread_worker(frame_to_process):
        """Background thread for OCR processing"""
        global is_processing, ocr_results
        is_processing = True
        
        try:
            print(f"  [Background] Processing frame...", end=" ", flush=True)
            process_timer = time.time()
            
            # Resize for faster processing
            small_frame = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
            
            # Convert BGR to RGB for PIL
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(small_frame_rgb)
            
            # Run Surya OCR (better accuracy than EasyOCR)
            predictions = recognition_predictor([pil_image], det_predictor=detection_predictor)
            
            elapsed = time.time() - process_timer
            print(f"Done in {elapsed:.2f}s")
            
            # Extract text from Surya results
            detected_texts = []
            if predictions and len(predictions) > 0:
                result = predictions[0]
                print(f"\n📝 Detected Text:")
                for line in result.text_lines:
                    text = line.text.strip() if hasattr(line, 'text') else ""
                    confidence = line.confidence if hasattr(line, 'confidence') else 0.9
                    if text:
                        print(f"  • {text} ({confidence:.1%})")
                        # Get bounding box
                        if hasattr(line, 'bbox'):
                            bbox = line.bbox  # (x1, y1, x2, y2) format
                            detected_texts.append((bbox, text, confidence))
                print()
            
            with process_lock:
                ocr_results = detected_texts
        except Exception as e:
            print(f"OCR Error: {e}")
        finally:
            is_processing = False
    
    while True:
        try:
            current_frame = frame_queue.get(timeout=0.5)
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed
                print(f"  [Background] FPS: {fps:.1f} | Detected text items: {len(ocr_results)}")
                fps_timer = time.time()
            
            # Start OCR in background thread every 5 frames
            if frame_count % 5 == 0 and not is_processing:
                ocr_thread = threading.Thread(target=ocr_thread_worker, args=(current_frame,), daemon=True)
                ocr_thread.start()
            
            # Display frame with OCR results
            display_frame = current_frame.copy()
            
            with process_lock:
                for bbox, text, confidence in ocr_results:
                    if bbox is not None:
                        # Scale bbox back up (was 0.5x downsampled)
                        # bbox format: (x1, y1, x2, y2)
                        x1, y1, x2, y2 = bbox
                        x1, y1, x2, y2 = int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Put text label
                        cv2.putText(display_frame, f"{text} ({confidence:.0%})", 
                                  (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Surya OCR Live", display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
                
        except queue.Empty:
            continue

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stop_event.set()
    cv2.destroyAllWindows()
