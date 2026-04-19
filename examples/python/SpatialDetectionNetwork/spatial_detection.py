#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import requests
import time
import threading
import base64
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 20

parser = argparse.ArgumentParser()
parser.add_argument(
    "--depthSource", type=str, default="stereo", choices=["stereo", "neural"]
)
parser.add_argument(
    "--serverUrl", type=str, default="http://127.0.0.1:8000", help="FastAPI server URL"
)
args = parser.parse_args()
# For better results on OAK4, use a segmentation model like "luxonis/yolov8-instance-segmentation-large:coco-640x480"
# for depth estimation over the objects mask instead of the full bounding box.
modelDescription = dai.NNModelDescription("yolov6-nano")
size = (640, 400)

if args.depthSource == "stereo":
    fps = STEREO_DEFAULT_FPS
else:
    fps = NEURAL_FPS

class SpatialVisualizer(dai.node.HostNode):
    def __init__(self, server_url="http://127.0.0.1:8000"):
        dai.node.HostNode.__init__(self)
        self.sendProcessingToPipeline(True)
        self.server_url = server_url
        self.current_detections = []
        self.last_send_time = time.time()
        self.send_interval = 0.1  # Send detections every 0.1 seconds
        self.last_frame_send_time = time.time()
        self.frame_send_interval = 0.1  # Send frames every 100ms
        
        # Create persistent session with connection pooling
        self.session = requests.Session()
        retry = Retry(connect=1, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    def build(self, depth:dai.Node.Output, detections: dai.Node.Output, rgb: dai.Node.Output):
        self.link_args(depth, detections, rgb) # Must match the inputs to the process method

    def process(self, depthPreview, detections, rgbPreview):
        depthPreview = depthPreview.getCvFrame()
        rgbPreview = rgbPreview.getCvFrame()
        depthFrameColor = self.processDepthFrame(depthPreview)
        self.displayResults(rgbPreview, depthFrameColor, detections.detections)

    def processDepthFrame(self, depthFrame):
        depthDownscaled = depthFrame[::4]
        if np.all(depthDownscaled == 0):
            minDepth = 0
        else:
            minDepth = np.percentile(depthDownscaled[depthDownscaled != 0], 1)
        maxDepth = np.percentile(depthDownscaled, 99)
        depthFrameColor = np.interp(depthFrame, (minDepth, maxDepth), (0, 255)).astype(np.uint8)
        return cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    def displayResults(self, rgbFrame, depthFrameColor, detections):
        height, width, _ = rgbFrame.shape
        
        # Collect detections data
        self.current_detections = []
        for detection in detections:
            self.drawBoundingBoxes(depthFrameColor, detection)
            self.drawDetections(rgbFrame, detection, width, height)
            
            # Store detection data
            self.current_detections.append({
                "label": detection.labelName,
                "confidence": detection.confidence * 100,
                "x": int(detection.spatialCoordinates.x),
                "y": int(detection.spatialCoordinates.y),
                "z": int(detection.spatialCoordinates.z)
            })
        
        # Send detections every 3 seconds
        current_time = time.time()
        if current_time - self.last_send_time >= self.send_interval:
            self.send_to_server()
            self.last_send_time = current_time
        
        # Send video frames more frequently (every 100ms)
        if current_time - self.last_frame_send_time >= self.frame_send_interval:
            self.send_video_frame(rgbFrame)
            self.last_frame_send_time = current_time
        
        # Print locally as well

        # Try to display (may fail on headless systems, which is OK)
        try:
            cv2.imshow("Color frame", rgbFrame)
            if cv2.waitKey(1) == ord('q'):
                self.stopPipeline()
        except cv2.error as e:
            # OpenCV display not available - continue without display
            # Frames are being sent to bridge.py for viewing
            pass

    def send_to_server(self):
        """Send current detections to the FastAPI server"""
        try:
            payload = {
                "timestamp": time.time(),
                "detections": self.current_detections
            }
            response = self.session.post(f"{self.server_url}/detections", json=payload, timeout=2)
            if response.status_code == 200:
                print(f"✓ Sent {len(self.current_detections)} detections to server")
            else:
                print(f"✗ Server error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to server. Make sure it's running on", self.server_url)
        except Exception as e:
            print(f"✗ Error sending to server: {e}")

    def send_video_frame(self, frame):
        """Send video frame to the FastAPI server"""
        try:
            # Encode frame to JPEG bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("✗ Failed to encode frame to JPEG")
                return
            
            # Encode to base64
            frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            payload = {"frame": frame_b64}
            response = self.session.post(f"{self.server_url}/video-frame", json=payload, timeout=1)
            if response.status_code == 200:
                print(f"✓ Sent video frame to server")
            else:
                print(f"✗ Video frame error: {response.status_code}")
        except requests.exceptions.Timeout:
            print("✗ Frame send timeout")
        except requests.exceptions.ConnectionError as e:
            print(f"✗ Cannot connect to server for video: {e}")
        except Exception as e:
            print(f"✗ Error sending video frame: {e}")

    def drawBoundingBoxes(self, depthFrameColor, detection):
        roiData = detection.boundingBoxMapping
        roi = roiData.roi
        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        cv2.rectangle(depthFrameColor, (int(topLeft.x), int(topLeft.y)), (int(bottomRight.x), int(bottomRight.y)), (255, 255, 255), 1)

    def drawDetections(self, frame, detection, frameWidth, frameHeight):
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)
        label = detection.labelName
        color = (255, 255, 255)
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

# Creates the pipeline and a default device implicitly
with dai.Pipeline() as p:
    # Define sources and outputs
    platform = p.getDefaultDevice().getPlatform()

    camRgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=fps)
    monoLeft = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    monoRight = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    if args.depthSource == "stereo":
        depthSource = p.create(dai.node.StereoDepth)
        depthSource.setExtendedDisparity(True)
        monoLeft.requestOutput(size).link(depthSource.left)
        monoRight.requestOutput(size).link(depthSource.right)
    elif args.depthSource == "neural":
        depthSource = p.create(dai.node.NeuralDepth).build(
            monoLeft.requestFullResolutionOutput(),
            monoRight.requestFullResolutionOutput(),
            dai.DeviceModelZoo.NEURAL_DEPTH_LARGE,
        )
    else:
        raise ValueError(f"Invalid depth source: {args.depthSource}")

    spatialDetectionNetwork = p.create(dai.node.SpatialDetectionNetwork).build(
        camRgb, depthSource, modelDescription
    )
    visualizer = p.create(SpatialVisualizer, args.serverUrl)

    spatialDetectionNetwork.spatialLocationCalculator.initialConfig.setSegmentationPassthrough(False)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    visualizer.build(
        spatialDetectionNetwork.passthroughDepth,
        spatialDetectionNetwork.out,
        spatialDetectionNetwork.passthrough,
    )

    print("Starting pipeline with depth source: ", args.depthSource)
    print("Sending detections to server: ", args.serverUrl)

    p.run()
