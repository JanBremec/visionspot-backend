from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel
from typing import List
import uvicorn
import subprocess
import threading
import time
import sys
import pickle
import cv2
import numpy as np
from pathlib import Path
from collections import deque
import requests
import json
import os
import warnings
import re
import asyncio
from queue import Queue

# Suppress setuptools pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
# Suppress torch pin_memory warning (Moondream data loading)
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

import face_recognition_models
print(face_recognition_models.__file__)

# Optional Geolocation (reverse geocoding)
try:
    from geopy.geocoders import Nominatim
    GEOLOCATION_AVAILABLE = True
    print("✓ Geolocation module loaded (geopy)")
except ImportError:
    GEOLOCATION_AVAILABLE = False
    print("⚠️  Geolocation module not found. Install with:")
    print("   pip install geopy")

# Optional IP Geolocation (to get laptop's current location)
try:
    import requests as geoip_requests
    IP_GEOLOCATION_AVAILABLE = True
    print("✓ IP Geolocation available (requests)")
except ImportError:
    IP_GEOLOCATION_AVAILABLE = False

# Optional Map visualization (Folium)
try:
    import folium
    from folium import plugins
    MAP_AVAILABLE = True
    print("✓ Map visualization (folium) loaded")
except ImportError:
    MAP_AVAILABLE = False
    print("⚠️  Map visualization not available. Install with:")
    print("   pip install folium")

# Optional OCR (EasyOCR)
try:
    import easyocr
    OCR_AVAILABLE = True
    print("✓ EasyOCR module loaded (models will be loaded on first use)")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  EasyOCR module not found. Install with:")
    print("   pip install easyocr")

# Optional face recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✓ face_recognition module loaded (models will be tested on first use)")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face_recognition module not found. Install with:")
    print("   pip install face-recognition")
    print("   pip install --default-timeout=1000 git+https://github.com/ageitgey/face_recognition_models")

# Optional Moondream image analysis
try:
    import moondream as md
    from PIL import Image
    MOONDREAM_AVAILABLE = True
    print("Loading Moondream model for image analysis...")
    moondream_model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI2ZjIxYjI4Ni04ODg4LTQyNWEtYmVhNy1mOGI4MDU1MmI3NjYiLCJvcmdfaWQiOiIyT0xnMmp6N1loOHc0NHFjdjE0czZ0SEJJNVJwTno0YiIsImlhdCI6MTc3NjQ0MjAxNywidmVyIjoxfQ.2GunKS_MLTKVlea33BWDW7FY-QFj0XQB9kT6F_c9EO0", local=False)
    print("✓ Moondream model loaded for image analysis")
except ImportError:
    MOONDREAM_AVAILABLE = False
    print("⚠️  Moondream module not found. Image analysis disabled. Install with:")
    print("   pip install moondream")

# Optional Voice Input/Output
try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    import edge_tts
    import asyncio
    import webrtcvad
    VOICE_AVAILABLE = True
    print("Loading Faster-Whisper for voice input...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")  # Lightweight for real-time
    tts_queue = Queue()  # Queue for TTS messages (ensures proper serialization)
    tts_lock = threading.Lock()  # Serialize TTS calls
    is_speaking = False  # True while TTS is playing — suppresses mic input to avoid feedback
    print("✓ Voice input/output available (Faster-Whisper + Edge-TTS)")
except ImportError as e:
    VOICE_AVAILABLE = False
    print("⚠️  Voice module not available. Install with:")
    print("   pip install sounddevice faster-whisper edge-tts webrtcvad")

# Prompt loading utility
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts folder"""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.md"
    if not prompt_path.exists():
        print(f"⚠️  Prompt file not found: {prompt_path}")
        return f"You are a helpful assistant for blind navigation."
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Remove markdown headers and keep just the core instruction
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        return '\n'.join(lines)
    except Exception as e:
        print(f"Error loading prompt {prompt_name}: {e}")
        return f"You are a helpful assistant for blind navigation."

def analyze_frame(frame):
    """Analyze a frame using Moondream to describe what's in it"""
    if not MOONDREAM_AVAILABLE or frame is None:
        return None
    
    try:
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Get description of what's in the image
        description = moondream_model.query(image, "Briefly describe what you see in this image in 1-2 sentences. Focus on main objects and layout.")["answer"]
        return description
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return None

def speak_text(text: str):
    """Queue text for TTS (non-blocking)"""
    if not VOICE_AVAILABLE or not text:
        return
    
    # Clean up text - remove excessive whitespace and problematic characters
    clean_text = ' '.join(text.split())
    # Replace em-dashes and other problematic punctuation
    clean_text = clean_text.replace('—', '. ')
    clean_text = clean_text.replace('  ', ' ')  # Remove double spaces created above
    
    # Add to queue - worker thread will process it
    print(f"📤 Queued TTS: {clean_text[:60]}... (queue size: {tts_queue.qsize()})")
    tts_queue.put(clean_text)

async def speak_text_async(text: str):
    """
    Speak text with minimal latency:
    - Streams all MP3 bytes from Edge TTS as fast as possible
    - Decodes entire buffer at once with miniaudio (very fast, <10ms)
    - Plays with sounddevice at low blocksize for snappy start
    """
    try:
        import io
        import miniaudio

        SAMPLE_RATE = 22050

        # Gather all MP3 bytes as fast as Edge TTS streams them
        mp3_buf = bytearray()
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural", rate="+20%")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.extend(chunk["data"])

        if not mp3_buf:
            print("⚠️ Edge TTS returned empty audio")
            return False

        # Decode MP3 → PCM (miniaudio is very fast, typically <10ms)
        decoded = miniaudio.decode(bytes(mp3_buf),
                                   output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=1, sample_rate=SAMPLE_RATE)
        samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0

        # Play with small blocksize = low startup latency (~23ms at 22050Hz)
        sd.play(samples, samplerate=SAMPLE_RATE, blocksize=512)
        sd.wait()
        return True

    except ImportError:
        print("⚠️ miniaudio not installed. Run: pip install miniaudio")
        return False
    except Exception as e:
        print(f"❌ TTS Playback Error: {str(e)[:120]}")
        return False

def tts_worker_thread():
    """Dedicated thread to process TTS queue using Edge-TTS
    
    Interrupts current playback when new messages arrive - plays latest immediately
    """
    if not VOICE_AVAILABLE:
        return
    
    print("🎤 TTS worker thread started")
    
    # Create event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def fetch_audio(text: str):
        """Fetch and decode TTS audio, return float32 numpy array or None"""
        try:
            import io
            import miniaudio
            SAMPLE_RATE = 22050
            mp3_buf = bytearray()
            communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural", rate="+20%")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buf.extend(chunk["data"])
            if not mp3_buf:
                return None, SAMPLE_RATE
            decoded = miniaudio.decode(bytes(mp3_buf),
                                       output_format=miniaudio.SampleFormat.SIGNED16,
                                       nchannels=1, sample_rate=SAMPLE_RATE)
            samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, SAMPLE_RATE
        except Exception as e:
            print(f"❌ TTS fetch error: {e}")
            return None, 22050

    while True:
        try:
            text = tts_queue.get()
            if text is None:
                break
            if not text or not text.strip():
                tts_queue.task_done()
                continue

            text_preview = text[:60] + "..." if len(text) > 60 else text
            print(f"🔊 Speaking ({len(text)} chars): {text_preview}")

            try:
                global is_speaking
                is_speaking = True

                # Split into sentences for streaming playback
                if len(text) > 200:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                else:
                    sentences = [text]

                # Pre-fetch pipeline: fetch sentence N+1 while playing sentence N
                prefetch_future = None
                prefetch_samples = None
                prefetch_sr = 22050

                for i, sentence in enumerate(sentences):
                    # Check if new messages arrived in queue - if so, interrupt and jump to latest
                    if tts_queue.qsize() > 0:
                        print(f"🛑 New message in queue ({tts_queue.qsize()} waiting) - interrupting current playback")
                        sd.stop()  # Stop current playback immediately
                        # Clear remaining messages and grab the latest one
                        while tts_queue.qsize() > 1:
                            tts_queue.get()  # Discard old messages
                            tts_queue.task_done()
                        # Mark current as done and let loop get the new message
                        is_speaking = False
                        tts_queue.task_done()
                        break
                    
                    # Start pre-fetching next sentence immediately
                    next_future = None
                    if i + 1 < len(sentences):
                        next_future = asyncio.ensure_future(
                            fetch_audio(sentences[i + 1]), loop=loop
                        )

                    # Get current sentence audio (from prefetch or fresh fetch)
                    if prefetch_future is not None:
                        samples, sr = loop.run_until_complete(
                            asyncio.wrap_future(prefetch_future) if not asyncio.isfuture(prefetch_future)
                            else prefetch_future
                        )
                    else:
                        samples, sr = loop.run_until_complete(fetch_audio(sentence))

                    # Play current sentence
                    if samples is not None:
                        sd.play(samples, samplerate=sr, blocksize=512)
                        sd.wait()

                    prefetch_future = next_future

                if is_speaking:  # Only print if we finished naturally (not interrupted)
                    print(f"✓ Finished speaking")

            except Exception as e:
                print(f"❌ TTS Error ({type(e).__name__}): {str(e)[:100]}")
            finally:
                is_speaking = False

            if tts_queue.qsize() > 0:
                print(f"📋 Queue status: {tts_queue.qsize()} messages waiting")
            
            # Only mark done if we didn't already in the interrupt logic
            try:
                tts_queue.task_done()
            except ValueError:
                pass  # Already marked done during interrupt

        except Exception as e:
            print(f"❌ TTS worker critical error: {e}")
            time.sleep(0.1)

    loop.close()

def voice_input_thread():
    """Continuously listen for voice input (non-blocking background thread)"""
    if not VOICE_AVAILABLE:
        print("⚠️  Voice input disabled - module not available")
        return
    
    try:
        import io
        from scipy.io import wavfile
        
        vad = webrtcvad.Vad()
        vad.set_mode(2)  # Less aggressive to reduce sensitivity to noise
        
        print("🎤 Voice input thread started - listening quietly...")
        
        # Suppress audio output feedback - loopback=False prevents echo
        with sd.InputStream(
            channels=1,
            samplerate=voice_config["sample_rate"],
            blocksize=voice_config["chunk_size"],
            dtype=np.int16,
            latency='high',
            device=None,  # Use default input device
        ) as stream:
            
            silence_count = 0
            is_recording = False
            accumulated_frames = []
            
            while True:
                try:
                    # Read audio data
                    audio_data, _ = stream.read(voice_config["chunk_size"])
                    audio_chunk = audio_data[:, 0].astype(np.int16).tobytes()
                    
                    # Check for voice activity (silently)
                    try:
                        is_speech = vad.is_speech(audio_chunk, voice_config["sample_rate"])
                    except:
                        is_speech = False
                    
                    # Suppress mic input while TTS is playing (prevents feedback loop)
                    if is_speaking:
                        is_recording = False
                        accumulated_frames = []
                        silence_count = 0
                        time.sleep(0.05)
                        continue
                    
                    if is_speech:
                        is_recording = True
                        silence_count = 0
                        accumulated_frames.append(audio_chunk)
                    elif is_recording:
                        silence_count += 1
                        silence_threshold = int(voice_config["silence_duration"] * voice_config["sample_rate"] / voice_config["chunk_size"])
                        
                        if silence_count > silence_threshold:
                            if accumulated_frames:
                                audio_bytes = b''.join(accumulated_frames)
                                transcribe_and_send_voice(audio_bytes)
                            
                            is_recording = False
                            accumulated_frames = []
                            silence_count = 0
                    
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.01)
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"⚠️ Voice input error (non-fatal): {type(e).__name__}")
                    time.sleep(0.1)
    
    except Exception as e:
        print(f"Failed to initialize voice input: {e}")
        print("💡 Tip: Voice input requires sounddevice. Install with: pip install sounddevice webrtcvad scipy")

def transcribe_and_send_voice(audio_bytes: bytes):
    """Transcribe audio and send as chat prompt to LLM with camera context (non-blocking)"""
    if not VOICE_AVAILABLE:
        return
    
    def process_voice():
        try:
            import io
            import wave
            import numpy as np

            # Wrap raw PCM int16 bytes in a proper WAV container for Whisper
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 bytes
                wf.setframerate(voice_config["sample_rate"])
                wf.writeframes(audio_bytes)
            wav_buffer.seek(0)

            # Transcribe with Faster-Whisper
            segments, info = whisper_model.transcribe(wav_buffer, language="en")
            transcription = " ".join([seg.text for seg in segments]).strip()

            if not transcription:
                return

            print(f"🎤 Transcribed: {transcription}")

            # Grab current camera frame for analysis
            with frame_lock:
                frame_copy = latest_frame.copy() if latest_frame is not None else None

            # Get latest detections immediately (don't wait for vision)
            with detection_lock:
                objects_copy = dict(objects_storage) if objects_storage else None
            with faces_lock:
                faces_copy = dict(faces_storage) if faces_storage else None

            # Analyze frame (may take a few seconds for Moondream API)
            frame_analysis = None
            if frame_copy is not None:
                frame_analysis = analyze_frame(frame_copy)
                if frame_analysis:
                    print(f"📷 Frame context: {frame_analysis}")

            # Build enriched user message with camera context
            if frame_analysis:
                enriched_message = f"{transcription}\n\n[Camera sees: {frame_analysis}]"
            else:
                enriched_message = transcription

            print(f"📤 Sending to LLM: {enriched_message[:80]}...")
            # Send to LLM as chat prompt
            send_to_llm_async(objects_copy, faces_copy, prompt_type="chat", user_message=enriched_message)

        except Exception as e:
            print(f"Voice transcription error: {e}")

    thread = threading.Thread(target=process_voice, daemon=True)
    thread.start()

# Load prompts on startup
EMERGENCY_PROMPT = load_prompt("emergency")
CHAT_PROMPT = load_prompt("chat")
LOCATION_PROMPT = load_prompt("location")
KNOWN_PERSON_PROMPT = load_prompt("known_person")
print("✓ System prompts loaded from /prompts folder")

app = FastAPI(title="Spatial Detection Server")

class Detection(BaseModel):
    label: str
    confidence: float
    x: int  # mm
    y: int  # mm
    z: int  # mm

class DetectionFrame(BaseModel):
    timestamp: float
    detections: List[Detection]

class VideoFrame(BaseModel):
    frame: str  # Base64 encoded frame

# Store for latest data
latest_detections = None
latest_frame = None
latest_frame_encoded = None
latest_llm_response = None  # Store latest LLM response for AI Log
latest_llm_timestamp = 0    # Timestamp of last LLM response
detection_history = deque(maxlen=10)  # Store last 10 detection frames
objects_storage = {}  # Store objects with normalized positions and z in meters
faces_storage = {}  # Store detected faces
chat_history = deque(maxlen=12)  # Store last 6 Q&A pairs (user + assistant = 12 messages) for chat mode only
known_face_encodings = []  # Face recognition encodings
known_face_names = []  # Face recognition names
known_person_greeting_timestamps = {}  # Track when we last greeted each known person
KNOWN_PERSON_GREETING_COOLDOWN = 30  # Seconds between greetings for the same person
last_emergency_timestamp = 0  # Track last emergency alert
EMERGENCY_COOLDOWN = 10  # Seconds between emergency alerts
detection_lock = threading.Lock()  # For incoming detections
faces_lock = threading.Lock()     # For face detection results (separate to avoid blocking)
ocr_lock = threading.Lock()       # For OCR results (separate to avoid blocking)
llm_lock = threading.Lock()       # For LLM response (to prevent race conditions)
chat_history_lock = threading.Lock()  # For chat history (to prevent race conditions)
frame_lock = threading.Lock()
frame_event = threading.Event()
stream_active = False
last_frame_time = 0
encoder_thread = None
object_id_counter = 0
face_process_counter = 0
face_detection_in_progress = False  # Skip if already processing
current_prompt_type = "emergency"  # Default to emergency prompt

# Face recognition settings
FACE_RECOGNITION_SETTINGS = {
    "process_every_n_frames": 8,      # Process every 8th frame (skip more frames for speed)
    "downscale_factor": 0.50,          # Downscale to 50%
    "tolerance": 0.6,                  # Face distance tolerance (lower = stricter)
    "model": "hog",                    # "hog" or "cnn" (cnn is slower but more accurate)
    "confidence_threshold": 0.5        # Only report matches above this threshold
}

face_detection_stats = {
    "frames_processed": 0,
    "faces_detected": 0,
    "last_check_time": 0
}

# OCR settings and storage
OCR_SETTINGS = {
    "check_interval": 5,               # Check for OCR every N seconds
    "downscale_factor": 0.5,           # Downscale to 50% for OCR processing
    "confidence_threshold": 0.3,       # Only report text with this confidence
    "language": "en"                   # Language for OCR
}

ocr_storage = {}  # Store OCR results
ocr_lock = threading.Lock()
ocr_reader = None  # EasyOCR reader (lazy-loaded)
last_ocr_time = 0
ocr_stats = {
    "frames_processed": 0,
    "text_detections": 0,
    "last_check_time": 0
}

# Location/GPS storage
location_storage = {
    "latitude": None,
    "longitude": None,
    "address": None,
    "street": None,
    "city": None,
    "country": None,
    "facility": None,
    "timestamp": None,
    "precision": None,  # GPS precision in meters
    "altitude": None
}
location_lock = threading.Lock()
geolocator = None  # Nominatim geocoder (lazy-loaded)
detection_history = deque(maxlen=50)  # Store last 50 detections with coordinates

# Voice input/output settings
voice_buffer = []  # Accumulate audio frames
voice_recording = False  # Flag to start/stop recording
voice_lock = threading.Lock()
voice_config = {
    "sample_rate": 16000,
    "chunk_size": 8192,  # Even larger buffer to handle TTS playback
    "silence_duration": 1.0,  # Seconds of silence to detect end of speech
    "vad_threshold": 0.5  # Voice activity detection threshold
}

def get_position_description(x_mm, y_mm):
    """Generate descriptive position text based on x,y coordinates in mm"""
    # Define thresholds for position description
    far_threshold = 200  # mm
    moderate_threshold = 100  # mm
    
    # Horizontal position
    if x_mm < -far_threshold:
        x_desc = "far left"
    elif x_mm < -moderate_threshold:
        x_desc = "left"
    elif x_mm < 0:
        x_desc = "slightly left"
    elif x_mm == 0:
        x_desc = "center horizontal"
    elif x_mm <= moderate_threshold:
        x_desc = "slightly right"
    elif x_mm <= far_threshold:
        x_desc = "right"
    else:
        x_desc = "far right"
    
    # Vertical position
    if y_mm < -far_threshold:
        y_desc = "far up"
    elif y_mm < -moderate_threshold:
        y_desc = "up"
    elif y_mm < 0:
        y_desc = "slightly up"
    elif y_mm == 0:
        y_desc = "center vertical"
    elif y_mm <= moderate_threshold:
        y_desc = "slightly down"
    elif y_mm <= far_threshold:
        y_desc = "down"
    else:
        y_desc = "far down"
    
    return x_desc, y_desc

def load_known_faces():
    """Load known face encodings from known_faces directory"""
    global known_face_encodings, known_face_names
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition disabled (module not installed)")
        return
    
    # Use absolute path relative to script location
    script_dir = Path(__file__).parent
    known_faces_dir = script_dir / "known_faces"
    
    # Handle nested known_faces structure
    if (known_faces_dir / "known_faces").is_dir():
        known_faces_dir = known_faces_dir / "known_faces"
    
    print(f"Loading known faces from '{known_faces_dir}'...")
    
    if not known_faces_dir.exists():
        print(f"⚠️  '{known_faces_dir}' folder not found!")
        print("Create it and add subfolders with person names containing their face images")
        print("Example structure:")
        print("  known_faces/")
        print("    ├── John/")
        print("    │   ├── photo1.jpg")
        print("    │   └── photo2.jpg")
        print("    └── Jane/")
        print("        └── photo1.jpg")
        return
    
    # Load from pickled encodings if available
    cache_file = known_faces_dir / ".face_cache.pkl"
    
    if cache_file.exists():
        print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
    else:
        # Scan folders for face images
        for person_name in os.listdir(known_faces_dir):
            person_path = known_faces_dir / person_name
            
            if person_path.is_dir():
                print(f"  Processing {person_name}...")
                
                for image_name in os.listdir(person_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                        image_path = str(person_path / image_name)
                        
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
    
    print(f"✓ Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} people\n")

def detect_faces_in_frame(frame):
    """Detect and recognize faces in a frame (runs in background thread)"""
    global faces_storage, FACE_RECOGNITION_AVAILABLE, face_detection_stats
    
    if not FACE_RECOGNITION_AVAILABLE or len(known_face_encodings) == 0:
        return {}
    
    try:
        # Optionally downscale for faster processing
        if FACE_RECOGNITION_SETTINGS["downscale_factor"] > 0:
            scale = FACE_RECOGNITION_SETTINGS["downscale_factor"]
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces and encodings
        face_locations = face_recognition.face_locations(rgb_frame, model=FACE_RECOGNITION_SETTINGS["model"])
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Compare faces
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, 
                face_encoding,
                tolerance=FACE_RECOGNITION_SETTINGS["tolerance"]
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
        
        # Store detected faces
        detected_faces = {}
        for i, (face_location, (name, confidence)) in enumerate(zip(face_locations, face_names)):
            detected_faces[i] = {
                "name": name,
                "confidence": confidence,
                "location": face_location
            }
            
            # Print detected face (only if confidence meets threshold)
            if name != "Unknown" and confidence >= FACE_RECOGNITION_SETTINGS["confidence_threshold"]:
                print(f"👤 Face detected: {name} ({confidence:.2%} confidence)")
            elif name == "Unknown":
                print(f"👤 Face detected: Unknown person")
        
        # Update stats (minimal lock time)
        face_detection_stats["frames_processed"] += 1
        face_detection_stats["faces_detected"] += len(detected_faces)
        
        # Print FPS every 30 processed frames
        if face_detection_stats["frames_processed"] % 30 == 0:
            elapsed = time.time() - face_detection_stats["last_check_time"] if face_detection_stats["last_check_time"] else 1
            fps = 30 / elapsed if elapsed > 0 else 0
            print(f"🎬 Face recognition: {fps:.1f} FPS | Total: {face_detection_stats['frames_processed']} frames, {face_detection_stats['faces_detected']} faces detected")
            face_detection_stats["last_check_time"] = time.time()
        
        # Use separate faces_lock (non-blocking for detections)
        with faces_lock:
            faces_storage = detected_faces
        
        return detected_faces
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return {}

def detect_faces_async(frame):
    """Run face detection in background thread (non-blocking, skip if already running)"""
    global face_detection_in_progress
    
    # Skip if already processing - never block the main thread
    if face_detection_in_progress:
        return
    
    def face_worker():
        global face_detection_in_progress
        try:
            face_detection_in_progress = True
            detect_faces_in_frame(frame)
        finally:
            face_detection_in_progress = False
    
    thread = threading.Thread(target=face_worker, daemon=True)
    thread.daemon = True
    thread.start()

def initialize_ocr_models():
    """Initialize EasyOCR reader on first use (lazy loading)"""
    global ocr_reader, OCR_AVAILABLE
    
    if not OCR_AVAILABLE or ocr_reader is not None:
        return
    
    try:
        print("🔄 Initializing EasyOCR models (this may take a minute on first run)...")
        ocr_reader = easyocr.Reader([OCR_SETTINGS["language"]], gpu=False)
        print("✓ EasyOCR models initialized!\n")
    except Exception as e:
        print(f"✗ Failed to initialize OCR models: {e}")
        OCR_AVAILABLE = False

def process_ocr_async(frame):
    """Process OCR in background thread (non-blocking)"""
    def ocr_worker():
        global ocr_storage, ocr_stats, OCR_AVAILABLE, ocr_reader
        
        if not OCR_AVAILABLE or ocr_reader is None:
            initialize_ocr_models()
            if ocr_reader is None:
                return
        
        try:
            # Downscale frame for faster processing
            scale = OCR_SETTINGS["downscale_factor"]
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Run EasyOCR
            results = ocr_reader.readtext(small_frame_rgb, detail=1)
            
            # Extract text results
            detected_texts = {}
            if results:
                idx = 0
                for (bbox, text, confidence) in results:
                    text = text.strip()
                    
                    if text and confidence >= OCR_SETTINGS["confidence_threshold"]:
                        # bbox is already in format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        # Convert to upscaled bounding box (x1, y1, x2, y2)
                        points = bbox
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)
                        
                        # Scale back up
                        bbox_scaled = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
                        
                        detected_texts[idx] = {
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox_scaled
                        }
                        idx += 1
                        print(f"📝 OCR: {text} ({confidence:.0%})")
            
            ocr_stats["frames_processed"] += 1
            ocr_stats["text_detections"] += len(detected_texts)
            
            with ocr_lock:
                ocr_storage = detected_texts
        
        except Exception as e:
            print(f"OCR error: {e}")
    
    thread = threading.Thread(target=ocr_worker, daemon=True)
    thread.start()

def initialize_geolocator():
    """Initialize geolocation on first use (lazy loading)"""
    global geolocator, GEOLOCATION_AVAILABLE
    
    if not GEOLOCATION_AVAILABLE or geolocator is not None:
        return
    
    try:
        print("🔄 Initializing Geolocation (Nominatim)...")
        geolocator = Nominatim(user_agent="depthai_bridge")
        print("✓ Geolocation initialized!\n")
    except Exception as e:
        print(f"✗ Failed to initialize geolocation: {e}")
        GEOLOCATION_AVAILABLE = False

def update_location(latitude, longitude):
    """Update current location and reverse geocode to get address (non-blocking)"""
    global geolocator, location_storage, GEOLOCATION_AVAILABLE
    
    if not GEOLOCATION_AVAILABLE:
        return
    
    def location_worker():
        try:
            if geolocator is None:
                initialize_geolocator()
            
            if geolocator is None:
                return
            
            # Reverse geocode coordinates with timeout
            try:
                location = geolocator.reverse(f"{latitude}, {longitude}", language='en', timeout=5)
                address = location.address if location else "Unknown"
            except Exception as geo_error:
                address = f"({latitude:.4f}, {longitude:.4f})"
            
            # Parse address to extract components
            parts = address.split(", ")
            street = parts[0] if len(parts) > 0 else None
            city = parts[-2] if len(parts) > 1 else None
            country = parts[-1] if len(parts) > 0 else None
            
            # Try to detect if it's a faculty/building
            facility = None
            for part in parts:
                if any(keyword in part.lower() for keyword in ['faculty', 'university', 'college', 'school', 'building', 'institute', 'lab', 'informatics']):
                    facility = part
                    break
            
            # Update storage with lock (non-blocking)
            with location_lock:
                location_storage.update({
                    "latitude": latitude,
                    "longitude": longitude,
                    "address": address,
                    "street": street,
                    "city": city,
                    "country": country,
                    "facility": facility,
                    "timestamp": time.time()
                })
            

        
        except Exception as e:
            # Silently handle errors - don't spam logs
            with location_lock:
                location_storage.update({
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": time.time()
                })
    
    # Run in background thread with daemon=True (non-blocking)
    thread = threading.Thread(target=location_worker, daemon=True)
    thread.daemon = True  # Ensure it doesn't block server shutdown
    thread.start()

def auto_update_location_from_gps():
    """Background thread that automatically updates location from hardcoded GPS coordinates (non-blocking)"""
    
    # Hardcoded location: Faculty of Informatics and Computer Science, Ljubljana, Slovenia
    HARDCODED_LAT = 46.05016775519865
    HARDCODED_LON = 14.469001196347097
    
    # Update location once on startup
    try:
        update_location(HARDCODED_LAT, HARDCODED_LON)
    except Exception:
        pass
    
    # Keep thread alive with minimal updates
    first_update = True
    while True:
        try:
            # Only update if it's been more than 60 seconds since last update
            with location_lock:
                last_update = location_storage.get("timestamp", 0)
            
            current_time = time.time()
            if first_update or (current_time - last_update) > 60:
                # Update location periodically (non-blocking, already in background)
                update_location(HARDCODED_LAT, HARDCODED_LON)
                first_update = False
            
            # Check every 5 seconds instead of 60
            time.sleep(5)
        except Exception:
            # Silently fail and retry
            time.sleep(5)

def generate_location_map():
    """Generate interactive map showing current location"""
    if not MAP_AVAILABLE:
        return None
    
    with location_lock:
        lat = location_storage.get("latitude")
        lon = location_storage.get("longitude")
        address = location_storage.get("address", "Unknown Location")
    
    if not lat or not lon:
        # Return default map centered on world if no location yet
        lat, lon = 20, 0
    
    # Create map centered on current location
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )
    
    # Add main location marker (person's position)
    folium.Marker(
        location=[lat, lon],
        popup=f"📍 You are here<br>{address}",
        tooltip="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa")
    ).add_to(m)
    
    # Add detection history to map
    with location_lock:
        for detection in detection_history:
            det_lat = detection.get("latitude")
            det_lon = detection.get("longitude")
            obj_name = detection.get("object", "Unknown")
            confidence = detection.get("confidence", 0)
            
            if det_lat and det_lon:
                # Determine color based on object type
                color = "red" if "person" in obj_name.lower() else "orange"
                
                folium.CircleMarker(
                    location=[det_lat, det_lon],
                    radius=5,
                    popup=f"{obj_name}<br>Confidence: {confidence:.1%}",
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)
    
    # Add zoom controls and other plugins
    folium.plugins.MiniMap().add_to(m)
    folium.plugins.Fullscreen().add_to(m)
    
    return m._repr_html_()

def send_to_llm_async(objects_dict, faces_dict=None, prompt_type="emergency", user_message=None):
    """Send object detections to LLM for blind person description (non-blocking)
    
    Args:
        objects_dict: Dictionary of detected objects (can be None/empty)
        faces_dict: Dictionary of detected faces (optional)
        prompt_type: Type of prompt to use ("emergency", "chat", "location")
        user_message: For chat mode, the actual user's message (optional)
    """
    def llm_request():
        global latest_llm_response, latest_llm_timestamp
        try:
            # Load system prompt dynamically (not cached) so changes take effect immediately
            system_content = load_prompt(prompt_type)
            if not system_content or system_content == f"You are a helpful assistant for blind navigation.":
                # Fallback if load fails
                system_prompts = {
                    "emergency": EMERGENCY_PROMPT,
                    "chat": CHAT_PROMPT,
                    "location": LOCATION_PROMPT,
                    "known_person": KNOWN_PERSON_PROMPT,
                }
                system_content = system_prompts.get(prompt_type, EMERGENCY_PROMPT)
            

            
            # Format objects for LLM (handle None/empty)
            objects_list = []
            if objects_dict:
                for idx, obj in objects_dict.items():
                    objects_list.append({
                        "name": obj["label"],
                        "confidence": f"{obj['confidence']:.1f}%",
                        "position": obj["position_description"],
                        "distance": f"{obj['z_m']:.2f}m"
                    })
            
            # Format faces for LLM
            faces_info = ""
            if faces_dict:
                faces_list = [f["name"] if f["name"] != "Unknown" else "unknown person" for f in faces_dict.values() if f.get("confidence", 0) > 0.5]
                if faces_list:
                    faces_info = f"\n\nPeople detected: {', '.join(faces_list)}."
            
            # Build user message based on prompt type
            if prompt_type == "emergency":
                user_msg = f"""Objects detected in front of me:

{json.dumps(objects_list, indent=2)}{faces_info}

Give me a quick, natural warning about what to do."""
            elif prompt_type == "location":
                objects_msg = f"Objects: {json.dumps(objects_list, indent=2)}" if objects_list else "No objects detected."
                user_msg = f"""I'm checking my surroundings:

{objects_msg}{faces_info}

Describe where I am and what the atmosphere is like around me."""
            elif prompt_type == "known_person":
                # For known person greeting, combine detected names with what they're doing
                # user_message contains the frame analysis (what they're doing)
                detected_names = faces_info.replace("\n\nPeople detected: ", "").replace(".", "").strip()
                if user_message:
                    user_msg = f"{detected_names} {user_message}"
                else:
                    user_msg = f"{detected_names} are here."
            else:  # chat
                # For chat mode, use the user's actual message if provided
                if user_message:
                    user_msg = user_message
                else:
                    # Fallback if no message provided
                    user_msg = "Hello, how are you?"
            

            
            # Build messages list based on prompt type
            if prompt_type == "chat":
                # For chat mode, include history (max 6 Q&A pairs = 12 messages)
                messages_list = [{"role": "system", "content": system_content}]
                with chat_history_lock:
                    for msg in chat_history:
                        messages_list.append(msg)
                messages_list.append({"role": "user", "content": user_msg})
            else:
                # For emergency/location, just single message
                messages_list = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_msg}
                ]
            
            response = requests.post(
                "https://server/v1/chat/completions",
                json={
                    "model": "my-model",
                    "messages": messages_list,
                    "stream": False
                },
                headers={"Content-Type": "application/json"},
                timeout=60  # Render free tier can be slow to wake up
            )
            
            if response.status_code == 200:
                try:
                    description = response.json()["choices"][0]["message"]["content"]
                    # Store the response and timestamp
                    with llm_lock:
                        latest_llm_response = description
                        latest_llm_timestamp = time.time()
                    
                    # For chat mode, add to history
                    if prompt_type == "chat" and user_message:
                        with chat_history_lock:
                            chat_history.append({"role": "user", "content": user_message})
                            chat_history.append({"role": "assistant", "content": description})
                    
                    print(f"\n🎙️ [{prompt_type.upper()}] {description}\n")
                    
                    # Speak the response (non-blocking)
                    speak_text(description)
                except Exception as e:
                    print(f"LLM JSON Error: {e}")
                    print(f"LLM Response: {response.text}")
            else:
                print(f"LLM HTTP Error: {response.status_code}")
                print(f"LLM Response: {response.text}")
        except requests.exceptions.Timeout:
            print(f"❌ LLM Timeout — server took >60s to respond (Render may be sleeping)")
            speak_text("Sorry, I didn't get a response. The server may be waking up, try again.")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ LLM Connection Error: {e}")
            speak_text("Sorry, I can't reach the server right now.")
        except Exception as e:
            print(f"❌ Error sending to LLM: {type(e).__name__}: {e}")
    
    # Run in background thread so it doesn't block
    thread = threading.Thread(target=llm_request, daemon=True)
    thread.start()

@app.post("/detections")
async def receive_detections(frame: DetectionFrame):
    """Endpoint to receive spatial detection data"""
    global latest_detections, object_id_counter, objects_storage
    
    with detection_lock:
        latest_detections = frame
        detection_history.append(frame)  # Add to history (keeps last 10)
        
        # Clear previous frame objects and store new ones
        objects_storage.clear()
        label_counts = {}  # Track count of each label for unique IDs
        

        
        for i, detection in enumerate(frame.detections):
            # Skip invalid detections (sensor errors where x=0, y=0, z=0)
            if detection.x == 0 and detection.y == 0 and detection.z == 0:
                continue
            
            # Normalize coordinates: x and y are already spatial (mm), z convert to meters
            x_mm = detection.x  # mm, already centered (middle is 0)
            y_mm = detection.y  # mm, already centered (middle is 0)
            z_m = detection.z / 1000.0  # Convert mm to meters
            
            # If z reading is 0m (sensor error), assume 1m instead
            if z_m == 0:
                z_m = 1.0
            
            # Create unique ID for objects with same label
            label_counts[detection.label] = label_counts.get(detection.label, 0) + 1
            unique_id = f"{detection.label}_{label_counts[detection.label]}" if label_counts[detection.label] > 1 or any(d.label == detection.label for d in frame.detections[:i]) else detection.label
            
            # Get descriptive position text
            x_desc, y_desc = get_position_description(x_mm, y_mm)
            
            # Store in dictionary by unique ID
            objects_storage[i] = {
                "label": detection.label,
                "unique_id": unique_id,
                "confidence": detection.confidence,
                "x_mm": x_mm,  # left (-) to right (+)
                "y_mm": y_mm,  # up (-) to down (+)
                "z_m": z_m,    # depth in meters
                "x_position": x_desc,
                "y_position": y_desc,
                "position_description": f"{x_desc}, {y_desc}"
            }
            
            # Store detection with location for map
            with location_lock:
                detection_record = {
                    "object": detection.label,
                    "confidence": detection.confidence,
                    "latitude": location_storage.get("latitude"),
                    "longitude": location_storage.get("longitude"),
                    "timestamp": time.time()
                }
                detection_history.append(detection_record)
            

        
        # Send to LLM ONLY if emergency condition is detected
        # For non-emergency, let the user control conversation via the interface
        if frame.detections:
            objects_copy = dict(objects_storage)
            
            # Find closest object distance
            closest_distance = min((obj["z_m"] for obj in objects_copy.values()), default=float('inf'))
            EMERGENCY_THRESHOLD = 2.0  # meters
            
            with faces_lock:
                faces_copy = dict(faces_storage)
                # Check if any known people are detected (not "Unknown")
                known_people = [face["name"] for face in faces_copy.values() if face.get("name") and face["name"] != "Unknown" and face.get("confidence", 0) > 0.5]
            
            # If something is close and it's a known person, greet instead of emergency
            if closest_distance < EMERGENCY_THRESHOLD:
                if known_people:
                    # Known person is close - greet them instead of emergency alert
                    current_time = time.time()
                    for person_name in known_people:
                        last_greeting = known_person_greeting_timestamps.get(person_name, 0)
                        if current_time - last_greeting > KNOWN_PERSON_GREETING_COOLDOWN:
                            known_person_greeting_timestamps[person_name] = current_time
                            
                            # Analyze current frame to describe what they're doing
                            with frame_lock:
                                frame_copy = latest_frame.copy() if latest_frame is not None else None
                            
                            frame_analysis = None
                            if frame_copy is not None:
                                frame_analysis = analyze_frame(frame_copy)
                            
                            send_to_llm_async(objects_copy, faces_copy, prompt_type="known_person", user_message=frame_analysis)
                else:
                    # Unknown object is close - trigger emergency (with cooldown)
                    global last_emergency_timestamp
                    current_time = time.time()
                    if current_time - last_emergency_timestamp > EMERGENCY_COOLDOWN:
                        last_emergency_timestamp = current_time
                        send_to_llm_async(objects_copy, faces_copy, prompt_type="emergency")
            
            # Also check for known people (even if not close) for greeting interrupt
            if known_people and closest_distance >= EMERGENCY_THRESHOLD:
                # Known person detected but not too close - still greet them
                current_time = time.time()
                for person_name in known_people:
                    last_greeting = known_person_greeting_timestamps.get(person_name, 0)
                    if current_time - last_greeting > KNOWN_PERSON_GREETING_COOLDOWN:
                        known_person_greeting_timestamps[person_name] = current_time
                        
                        # Analyze current frame to describe what they're doing
                        with frame_lock:
                            frame_copy = latest_frame.copy() if latest_frame is not None else None
                        
                        frame_analysis = None
                        if frame_copy is not None:
                            frame_analysis = analyze_frame(frame_copy)
                        
                        send_to_llm_async(objects_copy, faces_copy, prompt_type="known_person", user_message=frame_analysis)
    
    return {"status": "received", "count": len(frame.detections)}

@app.post("/video-frame")
async def receive_video_frame(data: dict):
    """Endpoint to receive video frames from camera"""
    global latest_frame, last_frame_time, face_process_counter, last_ocr_time
    
    try:
        import base64
        # Decode base64 frame
        frame_data = base64.b64decode(data.get("frame", ""))
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is not None:
            with frame_lock:
                latest_frame = frame
            last_frame_time = time.time()
            frame_event.set()
            
            # Detect faces based on configured frequency
            face_process_counter += 1
            if face_process_counter % FACE_RECOGNITION_SETTINGS["process_every_n_frames"] == 0:
                detect_faces_async(frame)
            
            # Run OCR check every N seconds (non-blocking)
            current_time = time.time()
            if OCR_AVAILABLE and (current_time - last_ocr_time) >= OCR_SETTINGS["check_interval"]:
                last_ocr_time = current_time
                process_ocr_async(frame)
        else:
            pass
    except Exception as e:
        print(f"✗ Error receiving video frame: {e}")
    
    return {"status": "received"}

@app.get("/latest")
async def get_latest_detections():
    """Get the latest detections"""
    with detection_lock:
        return latest_detections

@app.get("/last-detections")
async def get_last_detections():
    """Get the last 10 detected objects with their info"""
    with detection_lock:
        history_list = list(detection_history)
    
    return {
        "count": len(history_list),
        "detections": history_list
    }

@app.get("/objects")
async def get_stored_objects():
    """Get current stored objects with normalized coordinates"""
    with detection_lock:
        objects_copy = dict(objects_storage)
    
    return {
        "count": len(objects_copy),
        "objects": objects_copy
    }

@app.get("/faces")
async def get_detected_faces():
    """Get currently detected faces"""
    with faces_lock:
        faces_copy = dict(faces_storage)
    
    return {
        "count": len(faces_copy),
        "faces": faces_copy,
        "stats": {
            "frames_processed": face_detection_stats["frames_processed"],
            "faces_detected": face_detection_stats["faces_detected"]
        }
    }

@app.get("/face-recognition-settings")
async def get_face_settings():
    """Get current face recognition settings"""
    return FACE_RECOGNITION_SETTINGS

@app.get("/ocr")
async def get_ocr_results():
    """Get currently detected text from OCR"""
    with ocr_lock:
        ocr_copy = dict(ocr_storage)
    
    return {
        "count": len(ocr_copy),
        "text_regions": ocr_copy,
        "stats": {
            "frames_processed": ocr_stats["frames_processed"],
            "text_detections": ocr_stats["text_detections"]
        }
    }

@app.get("/ocr-settings")
async def get_ocr_settings():
    """Get current OCR settings"""
    return {
        "available": OCR_AVAILABLE,
        "settings": OCR_SETTINGS
    }

@app.get("/location")
async def get_location():
    """Get current location (coordinates + reverse geocoded address)"""
    with location_lock:
        location_copy = dict(location_storage)
    
    return location_copy

@app.post("/location")
async def set_location(latitude: float, longitude: float):
    """Update location and reverse geocode"""
    update_location(latitude, longitude)
    return {"status": "updating", "latitude": latitude, "longitude": longitude}

def encode_frames_worker():
    """Background thread that continuously encodes frames"""
    global latest_frame_encoded
    
    while True:
        try:
            with frame_lock:
                if latest_frame is not None:
                    frame_to_encode = latest_frame.copy()
                    
                    # Draw detected faces on the frame
                    with faces_lock:
                        faces_copy = dict(faces_storage)
                    
                    for face_idx, face_data in faces_copy.items():
                        try:
                            top, right, bottom, left = face_data["location"]
                            # No scaling needed - using full resolution now
                            
                            # Draw box
                            color = (0, 255, 0) if face_data["name"] != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame_to_encode, (left, top), (right, bottom), color, 2)
                            
                            # Draw label
                            if face_data["name"] == "Unknown":
                                label = "Unknown"
                            else:
                                label = f"{face_data['name']} ({face_data['confidence']:.2%})"
                            
                            cv2.rectangle(frame_to_encode, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            cv2.putText(frame_to_encode, label, (left + 6, bottom - 6),
                                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        except Exception:
                            pass
                    
                    resized = cv2.resize(frame_to_encode, (640, 480))
                    ret, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        latest_frame_encoded = buffer.tobytes()
            
            time.sleep(0.033)  # ~30 FPS encoding
        except Exception as e:
            print(f"Error in frame encoder: {e}")
            time.sleep(0.1)

def generate_mjpeg():
    """Generate MJPEG stream from latest frames"""
    consecutive_no_frame = 0
    placeholder_sent = False
    
    while stream_active:
        try:
            if latest_frame_encoded is not None:
                frame_bytes = latest_frame_encoded
                consecutive_no_frame = 0
                placeholder_sent = False
                
                # MJPEG boundary format
                yield b'--frame\r\n'
                yield b'Content-Type: image/jpeg\r\n'
                yield b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                yield frame_bytes
                yield b'\r\n'
            else:
                consecutive_no_frame += 1
                # Send placeholder if no frames available
                if not placeholder_sent:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for frames...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield b'--frame\r\n'
                        yield b'Content-Type: image/jpeg\r\n'
                        yield b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                        yield frame_bytes
                        yield b'\r\n'
                        placeholder_sent = True
                
                if consecutive_no_frame > 300:  # After 10 seconds of no frames, stop
                    print("No frames received for 10 seconds, stopping stream")
                    break
            
            time.sleep(0.033)  # ~30 FPS output
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Error in MJPEG: {e}")
            time.sleep(0.1)

@app.get("/frame")
async def get_frame():
    """Return the latest encoded frame as JPEG"""
    if latest_frame_encoded is None:
        # Return placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for frames...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        if ret:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
    return Response(content=latest_frame_encoded, media_type="image/jpeg")

@app.get("/video-stream")
async def video_stream():
    """Stream video frames as MJPEG"""
    global stream_active
    print(f"🎥 Stream client connected. Frames available: {latest_frame_encoded is not None}")
    stream_active = True
    try:
        return StreamingResponse(
            generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    finally:
        stream_active = False
        print("🎥 Stream client disconnected")

@app.get("/caretaker-data")
async def caretaker_data():
    """
    Aggregate endpoint for the caretaker dashboard.
    Returns a single JSON snapshot of all live data:
      - location (GPS coords, address, street)
      - objects  (current frame detections with position + depth)
      - faces    (recognized faces with confidence)
      - ocr      (detected text regions)
      - llm_response (latest AI/LLM response for the AI Log)
      - health   (stream status, last frame age)
    No existing endpoint or logic is modified; this is additive only.
    """
    # --- location ---
    with location_lock:
        loc = dict(location_storage)

    # --- objects (current frame) ---
    with detection_lock:
        objects_snapshot = {
            str(k): {
                "label":                v.get("label"),
                "unique_id":            v.get("unique_id"),
                "confidence":           v.get("confidence"),
                "x_mm":                 v.get("x_mm"),
                "y_mm":                 v.get("y_mm"),
                "z_m":                  v.get("z_m"),
                "x_position":           v.get("x_position"),
                "y_position":           v.get("y_position"),
                "position_description": v.get("position_description"),
            }
            for k, v in objects_storage.items()
        }

    # --- faces ---
    with faces_lock:
        faces_snapshot = {
            str(k): {
                "name":       v.get("name"),
                "confidence": v.get("confidence"),
            }
            for k, v in faces_storage.items()
        }

    # --- ocr ---
    with ocr_lock:
        ocr_snapshot = {
            str(k): {
                "text":       v.get("text"),
                "confidence": v.get("confidence"),
            }
            for k, v in ocr_storage.items()
        }

    # --- LLM response ---
    with llm_lock:
        llm_response = latest_llm_response
        llm_timestamp = latest_llm_timestamp * 1000 if latest_llm_timestamp else 0  # Convert to milliseconds for JS

    # --- health / stream ---
    now = time.time()
    frame_age = round(now - last_frame_time, 2) if last_frame_time else None

    return {
        "timestamp": now,
        "location": {
            "latitude":  loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "address":   loc.get("address"),
            "street":    loc.get("street"),
            "city":      loc.get("city"),
            "country":   loc.get("country"),
            "facility":  loc.get("facility"),
        },
        "objects":      objects_snapshot,
        "object_count": len(objects_snapshot),
        "faces":        faces_snapshot,
        "face_count":   len(faces_snapshot),
        "ocr":          ocr_snapshot,
        "ocr_count":    len(ocr_snapshot),
        "llm_response": llm_response,
        "llm_timestamp": llm_timestamp,
        "health": {
            "has_frames":         latest_frame is not None,
            "has_encoded_frames": latest_frame_encoded is not None,
            "stream_active":      stream_active,
            "last_frame_time":    last_frame_time,
            "frame_age_seconds":  frame_age,
        },
    }


@app.get("/caretaker")
async def caretaker_dashboard():
    """Serve the caretaker dashboard with live video stream"""
    caretaker_path = Path(__file__).parent / "caretaker (1).html"
    print(f"Looking for caretaker HTML at: {caretaker_path}")
    print(f"File exists: {caretaker_path.exists()}")
    
    if not caretaker_path.exists():
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error: Caretaker HTML file not found</h1>
            <p>Expected location: {caretaker_path}</p>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)
    
    try:
        with open(caretaker_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error loading caretaker dashboard</h1>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """
        print(f"Error loading caretaker dashboard: {e}")
        return HTMLResponse(content=error_html, status_code=500)

@app.get("/emergency-chat-test")
async def emergency_chat_test():
    """Serve the emergency chat test interface"""
    test_path = Path(__file__).parent / "emergency_chat_test.html"
    print(f"Looking for emergency chat test at: {test_path}")
    print(f"File exists: {test_path.exists()}")
    
    if not test_path.exists():
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error: Emergency Chat Test file not found</h1>
            <p>Expected location: {test_path}</p>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)
    
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error loading emergency chat test</h1>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """
        print(f"Error loading emergency chat test: {e}")
        return HTMLResponse(content=error_html, status_code=500)

@app.get("/viewer")
async def viewer():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spatial Detection Viewer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .video-container {
                display: flex;
                justify-content: center;
                margin: 20px 0;
                background-color: #000;
                border-radius: 4px;
                min-height: 480px;
                align-items: center;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }
            .loading {
                color: #888;
                font-size: 18px;
                text-align: center;
            }
            .status {
                text-align: center;
                padding: 10px;
                background-color: #e8f5e9;
                border-radius: 4px;
                margin-bottom: 20px;
                color: #2e7d32;
            }
            .status.error {
                background-color: #ffebee;
                color: #c62828;
            }
            .api-links {
                text-align: center;
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }
            .api-links a {
                display: inline-block;
                margin: 0 10px;
                color: #1976d2;
                text-decoration: none;
            }
            .api-links a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Spatial Detection Viewer</h1>
            <div class="status" id="status">
                <p id="status-text">Connecting to camera stream...</p>
            </div>
            <div class="video-container" id="video-container">
                <div class="loading">Loading stream...</div>
            </div>
            <div class="api-links">
                <p><strong>API Endpoints:</strong></p>
                <a href="/docs">📚 API Docs (Swagger)</a>
                <a href="/redoc">📖 ReDoc</a>
                <a href="/health">❤️ Health Check</a>
            </div>
        </div>
        
        <script>
            const container = document.getElementById('video-container');
            const statusDiv = document.getElementById('status');
            const statusText = document.getElementById('status-text');
            
            function loadStream() {
                const img = document.createElement('img');
                img.id = 'stream-img';
                img.alt = 'Live Stream';
                img.style.display = 'block';
                img.style.margin = '0 auto';
                img.style.maxWidth = '100%';
                img.style.height = 'auto';
                
                img.onload = function() {
                    statusDiv.className = 'status';
                    statusText.textContent = '✓ Stream connected';
                };
                
                img.onerror = function() {
                    statusDiv.className = 'status error';
                    statusText.textContent = '✗ Stream error - retrying...';
                    setTimeout(loadStream, 2000);
                };
                
                container.innerHTML = '';
                container.appendChild(img);
                
                // Force no-cache to get fresh frames
                const timestamp = Date.now();
                img.src = `/video-stream?t=${timestamp}`;
            }
            
            // Start loading stream
            loadStream();
            
            // Periodically reload to handle connection issues
            setInterval(() => {
                const img = document.getElementById('stream-img');
                if (img && !img.complete) {
                    // Still loading, don't interrupt
                    return;
                }
            }, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/map")
async def map_viewer():
    """Interactive map showing current location and detection history"""
    map_html = generate_location_map()
    
    if not map_html:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Location Map</title>
            <style>
                body { font-family: Arial; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f0f0f0; }
                .message { background: white; padding: 20px; border-radius: 8px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="message">
                <h2>Map Visualization Not Available</h2>
                <p>Install folium: <code>pip install folium</code></p>
            </div>
        </body>
        </html>
        """)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Location Map</title>
        <style>
            body {{ margin: 0; padding: 0; font-family: Arial; }}
            .container {{ display: flex; flex-direction: column; height: 100vh; }}
            .header {{ background: #2196F3; color: white; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .map-container {{ flex: 1; position: relative; }}
            .info {{ background: #f5f5f5; padding: 10px; border-top: 1px solid #ddd; font-size: 12px; }}
            .refresh-btn {{ position: absolute; top: 10px; right: 10px; z-index: 1000; padding: 8px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            .refresh-btn:hover {{ background: #45a049; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📍 Location Map - Real-time Position Tracking</h1>
            </div>
            <div class="map-container" id="mapContainer">
                {map_html}
            </div>
            <div class="info">
                <p>🔵 Blue marker: Your current position | 🔴 Red markers: Person detections | 🟠 Orange markers: Other objects</p>
                <p id="updateTime">Last update: {time.strftime('%H:%M:%S')}</p>
            </div>
        </div>
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        
        <script>
            // Auto-refresh map every 15 seconds
            setInterval(function() {{
                document.getElementById('mapContainer').innerHTML = '';
                location.reload();
            }}, 15000);
            
            // Update time display
            setInterval(function() {{
                const now = new Date();
                document.getElementById('updateTime').innerText = 'Last update: ' + now.toLocaleTimeString();
            }}, 1000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/health")
async def health_check():
    """Health check endpoint with diagnostics"""
    return {
        "status": "healthy",
        "has_frames": latest_frame is not None,
        "has_encoded_frames": latest_frame_encoded is not None,
        "last_frame_time": last_frame_time,
        "stream_active": stream_active,
        "detections": "yes" if latest_detections else "no"
    }

@app.get("/prompts")
async def get_available_prompts():
    """Get list of available LLM prompts and current active prompt"""
    available = []
    if PROMPTS_DIR.exists():
        for prompt_file in PROMPTS_DIR.glob("*.md"):
            available.append(prompt_file.stem)
    
    return {
        "available_prompts": sorted(available),
        "current_prompt": current_prompt_type,
        "descriptions": {
            "emergency": "Real-time obstacle avoidance warnings (default)",
            "chat": "Natural, friendly conversation about surroundings",
            "location": "Vivid description of current location and atmosphere"
        }
    }

@app.post("/set-prompt/{prompt_type}")
async def set_prompt(prompt_type: str):
    """Change the active LLM prompt type"""
    global current_prompt_type
    
    valid_prompts = ["emergency", "chat", "location"]
    if prompt_type not in valid_prompts:
        return {
            "status": "error",
            "message": f"Invalid prompt type. Choose from: {', '.join(valid_prompts)}"
        }
    
    current_prompt_type = prompt_type
    return {
        "status": "success",
        "current_prompt": current_prompt_type,
        "message": f"Switched to {prompt_type} prompt"
    }

@app.post("/clear-chat-history")
async def clear_chat_history():
    """Clear chat history (used for emergency mode entry or manual clear)"""
    with chat_history_lock:
        chat_history.clear()
    return {"status": "success", "message": "Chat history cleared"}

@app.post("/ask-llm/{prompt_type}")
async def ask_llm(prompt_type: str, body: dict = None):
    """Manually trigger LLM with specific prompt type using latest detections
    
    Optional body: {"user_message": "user's actual message"}
    """
    valid_prompts = ["emergency", "chat", "location"]
    if prompt_type not in valid_prompts:
        return {
            "status": "error",
            "message": f"Invalid prompt type. Choose from: {', '.join(valid_prompts)}"
        }
    
    print(f"\n[API] /ask-llm/{prompt_type} endpoint called")
    
    # Extract user message from body if provided
    user_message = None
    if body and isinstance(body, dict):
        user_message = body.get("user_message")
        if user_message:
            print(f"[API] User message: {user_message}")
    
    # For CHAT mode, analyze current frame and add description to user message
    if prompt_type == "chat":
        print(f"[API] Chat mode - analyzing frame...")
        with frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None
        
        if frame_copy is not None:
            # Analyze frame synchronously (blocking for this endpoint only)
            frame_analysis = analyze_frame(frame_copy)
            if frame_analysis:
                print(f"[API] Frame analysis: {frame_analysis}")
                # Add frame analysis to user message
                user_message = f"{user_message}\n\n[What I see right now: {frame_analysis}]"
                print(f"[API] Enhanced user message with frame analysis")
    
    with detection_lock:
        objects_copy = dict(objects_storage)
    with faces_lock:
        faces_copy = dict(faces_storage)
    
    print(f"[API] Objects: {len(objects_copy)}, Faces: {len(faces_copy)}")
    
    # Emergency mode requires detections
    if prompt_type == "emergency" and not objects_copy:
        return {
            "status": "warning",
            "message": "No objects detected. Nothing to alert about."
        }
    
    # Chat and location work with or without objects
    print(f"[API] Sending to LLM with {prompt_type} prompt...")
    send_to_llm_async(objects_copy if objects_copy else None, 
                      faces_copy if faces_copy else None, 
                      prompt_type=prompt_type,
                      user_message=user_message)
    
    return {
        "status": "success",
        "message": f"LLM request sent with {prompt_type} prompt",
        "objects_count": len(objects_copy),
        "faces_count": len(faces_copy)
    }

def run_spatial_detection(depth_source: str = "stereo"):
    """Run the spatial detection script in background"""
    script_path = Path(__file__).parent.parent / "examples" / "python" / "SpatialDetectionNetwork" / "spatial_detection.py"
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path), "--depthSource", depth_source],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read output from the script
        for line in process.stdout:
            print(f"[SpatialDetection] {line.strip()}")
        
        process.wait()
    except Exception as e:
        print(f"Error running spatial detection: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background threads on server startup"""
    global encoder_thread
    print("Starting Spatial Detection Server...")
    print("  🏥 Caretaker dashboard: http://127.0.0.1:8000/caretaker")
    print("  📺 Simple viewer: http://127.0.0.1:8000/viewer")
    print("  📚 API docs: http://127.0.0.1:8000/docs")
    
    # Load known faces for recognition
    load_known_faces()
    
    # Show face recognition settings
    print("\n⚙️  Face Recognition Settings:")
    print(f"  • Process every {FACE_RECOGNITION_SETTINGS['process_every_n_frames']} frame(s)")
    print(f"  • Downscale factor: {FACE_RECOGNITION_SETTINGS['downscale_factor']:.0%}")
    print(f"  • Detection model: {FACE_RECOGNITION_SETTINGS['model']}")
    print(f"  • Tolerance: {FACE_RECOGNITION_SETTINGS['tolerance']}")
    print(f"  • Confidence threshold: {FACE_RECOGNITION_SETTINGS['confidence_threshold']:.0%}\n")
    
    # Initialize OCR models (lazy loading in background)
    if OCR_AVAILABLE:
        print("⚙️  OCR (EasyOCR) Settings:")
        print(f"  • Check interval: {OCR_SETTINGS['check_interval']} seconds")
        print(f"  • Downscale factor: {OCR_SETTINGS['downscale_factor']:.0%}")
        print(f"  • Confidence threshold: {OCR_SETTINGS['confidence_threshold']:.0%}")
        print(f"  • Language: {OCR_SETTINGS['language']}")
        print("  • Status: Will load models on first text detection\n")
        # Initialize OCR in background thread (models are large)
        threading.Thread(target=initialize_ocr_models, daemon=True).start()
    else:
        print("⚠️  OCR disabled (EasyOCR not installed)\n")
    
    # Initialize geolocation (lazy loading)
    if GEOLOCATION_AVAILABLE:
        print("⚙️  Geolocation Settings:")
        print("  • Status: Will load on first location query")
        print("  • API endpoint: POST /location {latitude, longitude}\n")
        # Initialize in background thread
        threading.Thread(target=initialize_geolocator, daemon=True).start()
    
    # Start automatic GPS location update from laptop
    if IP_GEOLOCATION_AVAILABLE and GEOLOCATION_AVAILABLE:
        print("🛰️  Auto-GPS Enabled:")
        print("  • Updates every 30 seconds")
        print("  • Uses laptop's IP geolocation")
        print("  • Location displayed with each detection\n")
        threading.Thread(target=auto_update_location_from_gps, daemon=True).start()
    else:
        print("⚠️  Geolocation disabled (geopy not installed)\n")
    
    # Start frame encoder thread
    encoder_thread = threading.Thread(target=encode_frames_worker, daemon=True)
    encoder_thread.start()
    
    # Configuration: Set to False to disable voice input (if causing issues like pinging)
    ENABLE_VOICE_INPUT = True  # Change to False to disable voice commands
    
    # Start voice input thread (optional - disable if causing audio feedback)
    if VOICE_AVAILABLE and ENABLE_VOICE_INPUT:
        print("🎤 Voice Input Settings:")
        print("  • Input: Headphone microphone")
        print("  • Output: PC speaker")
        print("  • Model: Faster-Whisper (tiny)")
        print("  • Status: Listening for voice commands\n")
        threading.Thread(target=voice_input_thread, daemon=True).start()
        # Start dedicated TTS worker thread
        threading.Thread(target=tts_worker_thread, daemon=True).start()
    elif VOICE_AVAILABLE:
        print("🎤 Voice Input Settings:")
        print("  • Status: DISABLED (set ENABLE_VOICE_INPUT = True to enable)\n")
        # Still start TTS worker thread for text-to-speech output
        threading.Thread(target=tts_worker_thread, daemon=True).start()
    else:
        print("⚠️  Voice input disabled (pyaudio/faster-whisper not installed)\n")
    
    # Uncomment the line below to auto-start spatial detection script
    # threading.Thread(target=run_spatial_detection, daemon=True).start()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACIAL RECOGNITION & OCR BRIDGE SERVER")
    print("="*60)
    print("Starting FastAPI server on http://127.0.0.1:8000")
    print("  📺 Stream viewer: http://127.0.0.1:8000/viewer")
    print("  🗺️  Location map: http://127.0.0.1:8000/map")
    print("  🏥 Caretaker dashboard: http://127.0.0.1:8000/caretaker")
    print("  📚 API docs: http://127.0.0.1:8000/docs")
    print("  😊 Detected faces: http://127.0.0.1:8000/faces")
    print("  📝 OCR results: http://127.0.0.1:8000/ocr")
    print("  📍 Location: http://127.0.0.1:8000/location")
    print("  ⚙️  Settings: http://127.0.0.1:8000/face-recognition-settings")
    print("  ⚙️  OCR Settings: http://127.0.0.1:8000/ocr-settings")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, log_level="warning")