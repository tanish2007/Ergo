<p align="center">
  <h1 align="center">ERGO</h1>
  <p align="center"><b>Economical Responsive Ground Observant</b></p>
  <p align="center">
    Real-time distress detection and emergency guidance system powered by computer vision and AI
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Pose-red?logo=yolo" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/ElevenLabs-TTS-000000?logo=elevenlabs" alt="ElevenLabs">
  <img src="https://img.shields.io/badge/Raspberry_Pi-5-C51A4A?logo=raspberrypi&logoColor=white" alt="Raspberry Pi">
</p>

---

## Overview

**ERGO** is a real-time emergency response system that uses a Raspberry Pi 5 with a NoIR camera to continuously monitor a scene for distress postures. When a person is detected in distress (fallen, collapsed, hands on head, etc.), the system triggers an audible alarm and provides AI-generated first-aid guidance through a Bluetooth speaker — all hands-free.  

---

## How It Works

```
Raspberry Pi 5                          Laptop
+---------------------------+           +----------------------------------+
| NoIR Camera (MJPEG stream)|  --SSH--> | YOLOv8n-Pose (real-time)         |
| GPIO Button (trigger)     |  --SSH--> | Distress Detection Engine        |
| Bluetooth Speaker (output)|  <--SSH-- | Gemini 2.5 Flash (scene analysis)|
+---------------------------+           | ElevenLabs TTS (voice guidance)  |
                                        +----------------------------------+
```

### Detection Flow

```
MONITORING ──[5 distress frames]──> ALARM ──[distress clears]──> MONITORING
                                      │
                                      │ [button press]
                                      v
                                   PIPELINE
                                      │
                          ┌───────────┼───────────┐
                          v           v           v
                     Frame Saved   Gemini     ElevenLabs
                      as .jpg     Analysis   TTS -> MP3
                                      │
                                      v
                              Play guidance on
                              Bluetooth speaker
                                      │
                                      v
                                  MONITORING
```

1. **Continuous Monitoring** — Live MJPEG stream from Pi camera analyzed frame-by-frame with YOLOv8 Pose estimation
2. **Distress Detection** — 5 consecutive distress frames trigger the alarm (prevents false positives)
3. **Emergency Alarm** — Pre-recorded alert loops on the Bluetooth speaker
4. **Auto-Recovery** — Alarm stops automatically when distress posture clears
5. **Button Trigger** — Physical GPIO button activates the AI guidance pipeline
6. **AI Analysis** — Gemini 2.5 Flash analyzes the captured frame with pose context
7. **Voice Guidance** — ElevenLabs converts Gemini's medical instructions to natural speech
8. **Audio Playback** — Guidance plays through the Bluetooth speaker on the Pi

---

## Distress Postures Detected

| Check | Condition | Indicator |
|-------|-----------|-----------|
| Fall Detection | Head below hip level | Possible fall or collapse |
| Horizontal Torso | Shoulder-hip line is flat | Person lying down |
| Body Spread | Width-to-height ratio > 1.8 | On the ground |
| Hands Above Head | Wrists above nose | Distress signal |
| Face Covering | Both hands near face | Panic or pain response |
| Crouched/Curled | Compressed torso + head | Fetal position |
| Low in Frame | Body center below 75% of frame height | Collapsed |

---

## Anatomical Landmark Mapping

ERGO maps medical landmarks from pose keypoints for precise first-aid guidance:

| Landmark | Location | Use Case |
|----------|----------|----------|
| **Sternum** | Midpoint between shoulders | CPR compression target |
| **Outer Thigh** | Mid hip-to-knee | EpiPen injection site |
| **Neck/Carotid** | Between ear and shoulder | Pulse check |
| **Chest Center** | Between sternum and mid-hip | AED pad placement |

---

## Tech Stack

```mermaid
flowchart LR
    subgraph Pi [🟤 Raspberry Pi 5]
        CAM[📷 NoIR Camera]
        BTN[🔘 GPIO Button]
        SPK[🔊 Bluetooth Speaker]
    end

    subgraph Laptop [💻 Laptop / PC]
        OCV[🖼️ OpenCV]
        YOLO[🦴 YOLOv8n-Pose]
        GEM[✨ Gemini 2.5 Flash]
        TTS[🗣️ ElevenLabs TTS]
    end

    CAM -- MJPEG Stream --> OCV
    BTN -- GPIO Trigger --> GEM
    OCV --> YOLO
    YOLO -- Distress Detected --> GEM
    GEM -- First-Aid Text --> TTS
    TTS -- MP3 Audio --> SPK

    style Pi fill:#f9e2e2,stroke:#c51a4a,color:#000
    style Laptop fill:#e2ecf9,stroke:#4285f4,color:#000
```

---

## Project Structure

```
ERGO/
├── live_view.py          # Main application — state machine, alarm, live display
├── pose_detector.py      # YOLOv8 Pose distress detection + anatomical mapping
├── vision.py             # Gemini API integration for scene analysis
├── speech.py             # ElevenLabs TTS + Pi audio playback
├── pi_connection.py      # Persistent SSH/SFTP connection manager
├── pipeline.py           # Standalone pipeline (button mode / single-shot)
├── alarm.mp3             # Pre-generated emergency alert audio
├── yolov8n-pose.pt       # YOLOv8 Nano Pose model weights
├── requirements.txt      # Python dependencies
├── .env                  # API keys and Pi credentials (not committed)
└── captures/             # Saved frames, Gemini responses, and TTS audio
```

---

## Prerequisites

### Hardware

- **Raspberry Pi 5** with Raspberry Pi OS
- **Pi NoIR Camera Module** (or standard Pi Camera)
- **Bluetooth Speaker** (paired and connected via PulseAudio)
- **Push Button** connected to GPIO 17 (with pull-up resistor)
- **Laptop/PC** with Python 3.10+ and GPU recommended

### Raspberry Pi Setup

```bash
# Install audio and Bluetooth packages
sudo apt-get update
sudo apt-get install -y pulseaudio pulseaudio-module-bluetooth mpg123

# Start PulseAudio
pulseaudio --start

# Pair Bluetooth speaker
bluetoothctl scan on
bluetoothctl pair <MAC_ADDRESS>
bluetoothctl trust <MAC_ADDRESS>
bluetoothctl connect <MAC_ADDRESS>

# Set as default audio sink
pactl set-default-sink bluez_sink.<MAC_UNDERSCORED>.a2dp_sink
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aegis.git
cd aegis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install ultralytics opencv-python paramiko
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
PI_USER=your_pi_username
PI_HOST=your_pi_hostname.local
PI_PASSWORD=your_pi_password
```

### 4. Set Up Passwordless SSH (Recommended)

```bash
ssh-copy-id your_pi_username@your_pi_hostname.local
```

---

## Usage

### Live Monitoring Mode (Primary)

```bash
python live_view.py
```

Options:
```
--width    Frame width (default: 640)
--height   Frame height (default: 480)
--fps      Camera framerate (default: 15)
--pin      GPIO pin for button (default: 17)
--out-dir  Output directory (default: captures)
```

### Standalone Pipeline Mode

```bash
# Button-triggered mode (waits for GPIO press)
python pipeline.py --mode button

# Single-shot with existing image
python pipeline.py --mode once --image test_photo.jpg

# Text-only (no audio playback)
python pipeline.py --mode once --image test_photo.jpg --text-only
```

### Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit application |
| **GPIO Button** | Trigger Gemini + TTS pipeline |

---

## State Machine

| State | Description | Transitions |
|-------|-------------|-------------|
| `MONITORING` | Analyzing frames for distress | -> `ALARM` (5 consecutive distress frames) |
| `ALARM` | Emergency audio looping | -> `MONITORING` (distress clears) / -> `PIPELINE` (button press) |
| `PIPELINE` | Running Gemini analysis + TTS | -> `MONITORING` (pipeline complete) |

---

## Architecture Highlights

- **Persistent SSH Connections** — Single paramiko connection per component eliminates ~3s handshake overhead per operation
- **Edge-Cloud Hybrid** — Heavy ML inference on laptop, lightweight I/O on Pi
- **Bluetooth Audio via PulseAudio** — Wireless speaker support with automatic sink routing
- **Debounced Detection** — 5-frame consecutive threshold prevents false alarm triggers
- **Auto-Recovery** — Alarm automatically stops when the person is no longer in distress
- **5s Button Cooldown** — Prevents duplicate pipeline triggers from accidental presses
- **Pre-generated Alarm** — Static MP3 avoids API calls during emergencies

---

## API Rate Limits

| Service | Free Tier Limit | Retry Strategy |
|---------|----------------|----------------|
| Gemini 2.5 Flash | 20 requests/day | Auto-retry with 15s/30s/45s backoff |
| ElevenLabs | 10,000 chars/month | Pre-generated alarm to minimize usage |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Rudra Patel
