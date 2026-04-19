import cv2
import urllib.request
import numpy as np
import threading
import queue
import socket
import time
import os
import sys
from pathlib import Path
import pickle
import warnings

# Suppress setuptools pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import face_recognition

# Camera configuration - use bridge.py's video stream
# The bridge.py server should be running on localhost:8000
BRIDGE_URL = "http://localhost:8000"
CAMERA_URL = f"{BRIDGE_URL}/video-stream"

# Fallback to direct camera connection if bridge is not available
def get_camera_url():
    try:
        req = urllib.request.Request(CAMERA_URL, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        stream = urllib.request.urlopen(req, timeout=3)
        stream.close()
        print(f"✓ Connected to bridge camera stream")
        return CAMERA_URL
    except Exception as e:
        print(f"⚠️  Bridge not available ({e}), falling back to direct camera connection...")
        
        # Fallback: connect directly to camera
        base_url = "http://10.139.247.40:8080"
        endpoints = ["/video", "/mjpegfeed", "/stream", "/axis-cgi/mjpg/video.cgi", ""]
        
        for endpoint in endpoints:
            test_url = base_url + endpoint
            try:
                req = urllib.request.Request(test_url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                stream = urllib.request.urlopen(req, timeout=3)
                content_type = stream.headers.get('Content-Type', '')
                if 'mjpeg' in content_type.lower() or 'application/octet-stream' in content_type.lower():
                    stream.close()
                    return test_url
                stream.close()
            except Exception:
                pass
        
        return base_url + "/video"

url = get_camera_url()
print(f"Using camera: {url}\n")

# Load known faces
known_faces_dir = "known_faces"
# Handle nested known_faces structure
if os.path.isdir(os.path.join(known_faces_dir, "known_faces")):
    known_faces_dir = os.path.join(known_faces_dir, "known_faces")

print(f"Loading known faces from '{known_faces_dir}/'...")

known_face_encodings = []
known_face_names = []

if os.path.exists(known_faces_dir):
    # Load from pickled encodings if available
    cache_file = os.path.join(known_faces_dir, ".face_cache.pkl")
    
    if os.path.exists(cache_file):
        print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
    else:
        # Scan folders for face images
        for person_name in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_name)
            
            if os.path.isdir(person_path):
                print(f"  Processing {person_name}...")
                
                for image_name in os.listdir(person_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                        image_path = os.path.join(person_path, image_name)
                        
                        try:
                            # Load image and encode face
                            image = face_recognition.load_image_file(image_path)
                            face_encodings = face_recognition.face_encodings(image)
                            
                            if face_encodings:
                                known_face_encodings.append(face_encodings[0])
                                known_face_names.append(person_name)
                                print(f"    ✓ {image_name}")
                        except Exception as e:
                            print(f"    ✗ {image_name}: {str(e)[:50]}")
        
        # Cache the encodings
        if known_face_encodings: 
            with open(cache_file, 'wb') as f:
                pickle.dump((known_face_encodings, known_face_names), f)
else:
    print(f"\n⚠️  '{known_faces_dir}' folder not found!")
    print("Create it and add subfolders with person names containing their face images")
    print("Example structure:")
    print("  known_faces/")
    print("    ├── John/")
    print("    │   ├── photo1.jpg")
    print("    │   └── photo2.jpg")
    print("    └── Jane/")
    print("        └── photo1.jpg")

print(f"\n✓ Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} people")
print("\nStarting face recognition...")
print(f"Streaming from: {url}")
print("Press ESC to exit\n")
print("(Detections will print to console below)")
print("-" * 50)

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
        print(f"✓ Stream connected")
        
        # Get the boundary for multipart streams
        content_type = stream.headers.get('Content-Type', '')
        boundary = None
        if 'multipart' in content_type:
            # Extract boundary from Content-Type header
            for part in content_type.split(';'):
                if 'boundary=' in part:
                    boundary = part.split('boundary=')[1].strip().encode()
                    print(f"  Multipart stream with boundary: {boundary}")
                    break
        
        try:
            sock = stream.fp._sock
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (AttributeError, OSError):
            pass
        
        bytes_data = b''
        frames_received = 0
        bytes_read = 0
        
        while not stop_event.is_set():
            chunk = stream.read(32768)
            if not chunk:
                print(f"Stream ended (no more data)")
                break
            
            bytes_read += len(chunk)
            bytes_data += chunk
            
            # For multipart streams, look for boundaries
            if boundary and (b'--' + boundary) in bytes_data:
                # Extract content between boundaries
                parts = bytes_data.split(b'--' + boundary)
                for part in parts[1:]:  # Skip first empty part
                    # Look for JPEG data in this part
                    a = part.find(b'\xff\xd8')
                    b = part.find(b'\xff\xd9', a)
                    
                    if a != -1 and b != -1:
                        jpg = part[a:b+2]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            frames_received += 1
                            if frames_received == 1:
                                print(f"✓ First frame received ({frame.shape})")
                            try:
                                frame_queue.put_nowait(frame)
                            except queue.Full:
                                pass
                
                # Keep data after last boundary
                if (b'--' + boundary) in bytes_data:
                    bytes_data = bytes_data.split(b'--' + boundary)[-1]
            else:
                # For regular MJPEG streams without boundary
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9', a)
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        frames_received += 1
                        if frames_received == 1:
                            print(f"✓ First frame received ({frame.shape})")
                        try:
                            frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass
        
        print(f"Stream closed after {frames_received} frames ({bytes_read} bytes read)")
        stream.close()
    except Exception as e:
        print(f"✗ Stream error: {e}")

# Start camera thread
thread = threading.Thread(target=read_stream, daemon=True)
thread.start()

# Wait a bit and check if we're getting frames from bridge
time.sleep(2)
if frame_queue.empty():
    print("\n⚠️  Bridge stream connected but NOT receiving frames!")
    print("    (The bridge streams frames from what it receives)")
    print("\n    To use bridge as camera source, you need to:")
    print("    1. Run a camera source that posts frames to http://localhost:8000/video-frame")
    print("    2. Or use direct camera connection")
    print("\nFalling back to direct camera...")
    stop_event.set()
    time.sleep(0.5)
    
    # Reset and try direct camera
    stop_event.clear()
    direct_url = "http://10.139.247.40:8080"
    
    # Try to find working endpoint
    endpoints = ["/video", "/mjpegfeed", "/stream", "/axis-cgi/mjpg/video.cgi", ""]
    found_url = False
    
    for endpoint in endpoints:
        test_url = direct_url + endpoint
        try:
            print(f"  Testing {endpoint}...")
            req = urllib.request.Request(test_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            stream = urllib.request.urlopen(req, timeout=3)
            content_type = stream.headers.get('Content-Type', '')
            print(f"    Content-Type: {content_type}")
            if 'mjpeg' in content_type.lower() or 'application/octet-stream' in content_type.lower():
                stream.close()
                print(f"  ✓ Found working endpoint: {endpoint}")
                url = test_url
                found_url = True
                break
            stream.close()
        except Exception as e:
            print(f"    Failed: {str(e)[:50]}")
    
    if found_url:
        print(f"Using fallback camera: {url}\n")
        thread = threading.Thread(target=read_stream, daemon=True)
        thread.start()
        time.sleep(2)
    else:
        print("\n✗ No camera sources available!")
        print("\nTo use this script, you need one of:")
        print("  1. Bridge streaming (run a camera source posting to http://localhost:8000/video-frame)")
        print("  2. Direct camera at http://10.139.247.40:8080")
        print("  3. Modify BRIDGE_URL or camera endpoints in this script")
        sys.exit(1)

print()

try:
    fps_timer = 0
    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame for speed
    
    # Store face data separately
    face_locations = []
    face_names = []
    
    while True:
        try:
            current_frame = frame_queue.get(timeout=0.5)
            frame_count += 1
            
            # Process faces every N frames
            if frame_count % process_every_n_frames == 0:
                # Resize for faster processing
                small_frame = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces and encodings
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Compare faces
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, 
                        face_encoding,
                        tolerance=0.6
                    )
                    name = "Unknown"
                    confidence = 0
                    
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, 
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                    
                    face_names.append((name, confidence))
            
            # Display frame
            display_frame = current_frame.copy()
            
            if len(face_locations) > 0:
                for (top, right, bottom, left), (name, confidence) in zip(
                    face_locations, 
                    face_names
                ):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label
                    if name == "Unknown":
                        label = "Unknown"
                    else:
                        label = f"{name} ({confidence:.2f})"
                    
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display_frame, label, (left + 6, bottom - 6), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Console output
                    if name == "Unknown":
                        print(f"👤 Face detected: Unknown")
                    else:
                        print(f"👤 Face detected: {name} ({confidence:.2%})")
            
            # Show info
            cv2.putText(display_frame, "ESC to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            try:
                cv2.imshow("Face Recognition", display_frame)
            except Exception as e:
                print(f"Display error: {e}")
            
            # Show FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer if fps_timer else 1
                fps = 30 / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f} | Processing every {process_every_n_frames} frames")
                fps_timer = time.time()
            
            try:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"Waitkey error (display may not be available): {e}")
                # Continue running even if display is unavailable
                time.sleep(0.001)
        
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
