#!/usr/bin/env python3
"""Test camera and bridge connectivity"""
import urllib.request
import socket
import sys

def test_url(url, name="URL"):
    """Test if a URL is reachable and what it returns"""
    print(f"\n📍 Testing {name}: {url}")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Connection': 'close'
        })
        response = urllib.request.urlopen(req, timeout=5)
        content_type = response.headers.get('Content-Type', 'unknown')
        content_length = response.headers.get('Content-Length', 'unknown')
        
        # Read first chunk
        first_chunk = response.read(1024)
        response.close()
        
        print(f"  ✓ Connected!")
        print(f"    Status: {response.status}")
        print(f"    Content-Type: {content_type}")
        print(f"    Content-Length: {content_length}")
        print(f"    First 100 bytes: {first_chunk[:100]}")
        return True
    except socket.timeout:
        print(f"  ✗ Timeout (server not responding)")
        return False
    except ConnectionRefusedError:
        print(f"  ✗ Connection refused (server not running?)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_host(host, port, name="Host"):
    """Test if a host:port is reachable"""
    print(f"\n🔌 Testing {name}: {host}:{port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"  ✓ Host reachable on port {port}")
            return True
        else:
            print(f"  ✗ Host not reachable on port {port}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

print("=" * 60)
print("CAMERA & BRIDGE CONNECTIVITY TEST")
print("=" * 60)

# Test bridge
print("\n--- BRIDGE (localhost:8000) ---")
bridge_host = test_host("localhost", 8000, "Bridge")
if bridge_host:
    test_url("http://localhost:8000/video-stream", "Bridge /video-stream")
    test_url("http://localhost:8000/last-detections", "Bridge /last-detections")

# Test camera
print("\n--- CAMERA (10.139.247.40:8080) ---")
camera_host = test_host("10.139.247.40", 8080, "Camera")
if camera_host:
    for endpoint in ["/video", "/mjpegfeed", "/stream", "/axis-cgi/mjpg/video.cgi"]:
        test_url(f"http://10.139.247.40:8080{endpoint}", f"Camera {endpoint}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if not bridge_host and not camera_host:
    print("❌ Neither bridge nor camera is reachable!")
    print("   → Start bridge.py on localhost:8000")
    print("   → Check if camera server is running on 10.139.247.40:8080")
elif bridge_host and not camera_host:
    print("⚠️  Bridge is running, but camera is unreachable")
    print("   → Check camera IP and port")
    print("   → Bridge may need frames sent to it")
elif not bridge_host and camera_host:
    print("⚠️  Camera is running, but bridge is not")
    print("   → Start bridge.py if you want to use it")
    print("   → Or use direct camera connection")
else:
    print("✓ Both bridge and camera are reachable!")
