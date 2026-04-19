import cv2
import urllib.request
import numpy as np
import threading
import queue
import socket
import time
import moondream as md
from PIL import Image
import io

# Initialize moondream model
print("Loading moondream model...")
model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI2ZjIxYjI4Ni04ODg4LTQyNWEtYmVhNy1mOGI4MDU1MmI3NjYiLCJvcmdfaWQiOiIyT0xnMmp6N1loOHc0NHFjdjE0czZ0SEJJNVJwTno0YiIsImlhdCI6MTc3NjQ0MjAxNywidmVyIjoxfQ.2GunKS_MLTKVlea33BWDW7FY-QFj0XQB9kT6F_c9EO0", local=False)
print("Model loaded!\n")

# Try common IP Webcam endpoints
base_url = "http://100.86.122.106:8080"
endpoints = [
    "/video",           # IP Webcam
    "/mjpegfeed",       # Iriun & others
    "/stream",          # Generic
    "/axis-cgi/mjpg/video.cgi",  # Axis cameras
    "",                 # Root (fallback)
]

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
    except Exception as e:
        pass

if not url:
    url = base_url + "/video"

print(f"Using URL: {url}\n")
print("Starting camera analyzer...")
print("Press ENTER to analyze current frame")
print("Press ESC to exit\n")

# Frame queue for threading (keep only latest frame)
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
analysis_lock = threading.Lock()
current_analysis = None

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
            
            # Find JPEG frame boundaries
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9', a)
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode frame
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    try:
                        frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
        
        stream.close()
    except Exception as e:
        print(f"Stream error: {e}")

def analyze_frame(frame):
    """Analyze frame with moondream"""
    try:
        print("\n🔍 Analyzing image...")
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Get caption
        caption = model.caption(image)["caption"]
        
        # Ask specific question
        answer = model.query(image, "What's in this image? Answer in 2-3 sentences. Focus on main features and objects")["answer"]
        
        result = f"Caption: {caption}\n\nAnalysis: {answer}"
        print(result)
        print("\n" + "="*50)
        
        return result
    except Exception as e:
        error_msg = f"Error analyzing image: {e}"
        print(error_msg)
        return error_msg

# Start background reader thread
thread = threading.Thread(target=read_stream, daemon=True)
thread.start()

try:
    fps_timer = time.time()
    frame_count = 0
    current_frame = None
    
    while True:
        try:
            # Get latest frame from queue
            current_frame = frame_queue.get(timeout=0.5)
            
            # Display frame with analysis if available
            display_frame = current_frame.copy()
            
            with analysis_lock:
                if current_analysis:
                    # Add text overlay
                    text = "Press ENTER to analyze | ESC to exit"
                    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera Analyzer", display_frame)
            frame_count += 1
            
            # Show FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                fps = 30 / elapsed
                print(f"FPS: {fps:.1f}")
                fps_timer = time.time()
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\nExiting...")
                break
            elif key == 13:  # ENTER
                if current_frame is not None:
                    with analysis_lock:
                        current_analysis = analyze_frame(current_frame)
        
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

except KeyboardInterrupt:
    pass
finally:
    stop_event.set()
    cv2.destroyAllWindows()
    print("Closed")
