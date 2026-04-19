# VisionSpot - AI-Powered Blind Navigation Assistant

Real-time AI navigation system for blind users using OAK camera hardware, facial recognition, OCR, and LLM integration.

> **Project Overview**: This project uses Luxonis DepthAI hardware for real-time spatial detection, combined with Python ML models and LLM APIs to create an intelligent navigation guide.

## 🎯 What This Project Does

VisionSpot is an AI assistant that helps blind users navigate by:
- **Real-time obstacle detection** - Warns about objects within 2 meters
- **Spatial awareness** - Describes left/right/up/down positioning
- **Facial recognition** - Identifies known people nearby
- **Text detection (OCR)** - Reads signs and text in real-time
- **Voice interaction** - Listens to voice commands, responds with audio
- **Location awareness** - Reverse geocodes GPS to address
- **Caretaker dashboard** - Real-time monitoring for caregivers

## 🏗️ How We Used DepthAI

### The OAK Camera Pipeline

**DepthAI handles:**
1. **Stereo depth estimation** - Calculates distance to objects (spatial coordinates)
2. **YOLOv6 neural inference** - Detects objects in real-time
3. **Synchronized RGB + Depth** - Provides color video + depth map

**Our implementation (`examples/python/SpatialDetectionNetwork/spatial_detection.py`):**
```python
# DepthAI pipeline setup
modelDescription = dai.NNModelDescription("yolov6-nano")  # Object detection
pipeline = dai.Pipeline()

# Nodes for stereo depth + neural detection
spatialDetectionNode = pipeline.createYoloSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()   # Left camera
monoRight = pipeline.createMonoCamera()  # Right camera
stereo = pipeline.createStereoDepth()    # Depth calculation
rgb = pipeline.createColorCamera()       # RGB feed

# Link nodes together
monoLeft.out → stereo.left
monoRight.out → stereo.right
stereo.depth → spatialDetectionNode.inputDepth
rgb.preview → spatialDetectionNode.input
```

### Why DepthAI?

- **Hardware acceleration**: Neural inference runs on device (not PC)
- **Stereo depth**: Embedded stereo pair calculates distance without extra setup
- **Real-time**: <100ms latency for detection + depth
- **Compact**: Works with standard OAK camera (4K RGB + stereo cameras)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install depthai opencv-python numpy requests fastapi uvicorn
pip install face-recognition easyocr geopy folium
pip install faster-whisper edge-tts sounddevice webrtcvad
```

### 2. Start Camera Input (DepthAI)

```bash
cd examples/python/SpatialDetectionNetwork
python spatial_detection.py --serverUrl http://127.0.0.1:8000
```

This sends object detections every **0.1 seconds** to the backend.

### 3. Start Backend Server

```bash
cd backend
python bridge.py
```

Server runs at: `http://127.0.0.1:8000`

### 4. Open Dashboard

Visit: `http://127.0.0.1:8000/caretaker`

## 📊 Data Flow

```
┌─────────────────────────┐
│   OAK Camera (DepthAI)  │
│  - YOLOv6 detection     │
│  - Stereo depth         │
│  - RGB video            │
└────────────┬────────────┘
             │ (HTTP POST @ 0.1s)
             ↓
┌─────────────────────────┐
│  bridge.py (FastAPI)    │
│  /detections endpoint   │
└────────────┬────────────┘
             │
      ┌──────┴──────┬──────────┬──────────┐
      ↓             ↓          ↓          ↓
   Objects      Faces       OCR       Location
   [Filter]  [Recognition] [Text]   [Geocode]
      │             │          │          │
      └──────────────┼──────────┴──────────┘
                     ↓
            ┌─────────────────┐
            │  LLM Backend    │
            │ (Emergency msg) │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │  TTS (Edge-TTS) │
            │  Voice Output   │
            └─────────────────┘
```

## 🔧 Key Features

### Real-Time Detection
- **Detection interval**: 0.1 seconds (100ms)
- **Emergency threshold**: 2.0 meters
- **Processing speed**: 8 FPS for faces, continuous for objects

### Face Recognition
- **Downscale**: 50% for speed
- **Tolerance**: 0.6 (strict matching)
- **Model**: HOG detector
- **Known faces**: Stored in `backend/known_faces/`

### Voice System
- **Input**: Faster-Whisper (speech-to-text)
- **Output**: Edge-TTS (text-to-speech)
- **Interrupt**: New messages interrupt current speech immediately

### Smart Prompts
Never says "2.5 meters" - instead says:
- "very close" (< 1m)
- "nearby" (1-2m) 
- "not far" (2-3m)
- "up ahead" (far)

## 📁 Project Structure

```
depthai-core/
├── backend/
│   ├── bridge.py                    # Main FastAPI server
│   ├── prompts/                     # AI system prompts
│   ├── known_faces/                 # Face recognition training data
│   └── caretaker (1).html          # Monitoring dashboard
│
└── examples/python/SpatialDetectionNetwork/
    └── spatial_detection.py         # DepthAI camera → object detection
```

## 🎮 API Endpoints

- `POST /detections` - Receive DepthAI detections
- `GET /objects` - Current detected objects
- `GET /faces` - Recognized people
- `GET /ocr` - Detected text
- `GET /location` - GPS + address
- `POST /ask-llm/emergency` - Trigger emergency response
- `GET /video-stream` - Live MJPEG feed
- `GET /caretaker-data` - All data as JSON
- `GET /docs` - Interactive API documentation

See [backend/README.md](backend/README.md) for full documentation.

## 🛠️ Configuration

### Emergency Alert Distance
`backend/bridge.py` line ~1244:
```python
EMERGENCY_THRESHOLD = 2.0  # meters
```

### Detection Frequency
`examples/python/SpatialDetectionNetwork/spatial_detection.py` line ~48:
```python
self.send_interval = 0.1  # 100ms (detections per second)
```

### Face Recognition Settings
`backend/bridge.py` line ~607:
```python
FACE_RECOGNITION_SETTINGS = {
    "downscale_factor": 0.50,      # 50% speed
    "process_every_n_frames": 8,   # Every 8th frame
    "tolerance": 0.6,              # Strict matching
}
```

## 🔌 DepthAI Hardware Setup

**What you need:**
- Luxonis OAK camera (OAK-D, OAK-D Pro, etc.)
- USB 3.0+ cable (OAK-D USB version)
- Windows/Linux/macOS with USB support

**Installation:**
```bash
pip install depthai
```

**Test connection:**
```bash
python -c "import depthai as dai; print('✓ DepthAI connected')"
```

## 📊 Monitoring

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### All Live Data
```bash
curl http://127.0.0.1:8000/caretaker-data | jq
```

## 🐛 Troubleshooting

### Camera Not Found
```bash
# Check DepthAI installation
python -c "import depthai as dai; dai.Device()"

# List connected devices
python -c "import depthai as dai; print(dai.Device.getDeviceByMxId(dai.Device.getFirstAvailableDevice()))"
```

### Slow Detection
- Reduce face recognition frequency: `process_every_n_frames: 16`
- Lower downscale factor: `downscale_factor: 0.25`

### Connection Issues
- Ensure OAK camera is plugged in USB 3.0+ port
- Try different USB port
- Restart camera: `systemctl restart depthai` (Linux)

## 📚 Resources

- **DepthAI Docs**: https://docs.luxonis.com/
- **YOLOv6**: Object detection model
- **Faster-Whisper**: Speech recognition
- **FastAPI**: Web framework

## 👥 Team

- **Development**: Jan Bremec (jan04bremec@gmail.com)
- **Original Concept**: Lea Vodopivec

---

**Last Updated**: April 19, 2026
