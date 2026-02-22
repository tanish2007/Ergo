#!/usr/bin/env python3
"""
Guardian Eye — Live YOLO Pose + Emergency Alarm + Gemini Pipeline

Flow:
  1. Live camera feed from Pi -> YOLO Pose on every frame
  2. Distress detected (5 consecutive frames) -> alarm loops on Pi earphone
  3. GPIO button press on Pi -> stop alarm -> Gemini analysis -> TTS
  4. Play Gemini guidance once on earphone -> resume monitoring

Press Q or ESC to quit.
"""

import argparse
import cv2
import numpy as np
import os
import paramiko
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pose_detector import (
    _load_model, _kp, _mid, _get_anatomical_sites,
    KP_CONF,
    NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)

PI_USER = os.getenv("PI_USER", "stackunderflow")
PI_HOST = os.getenv("PI_HOST", "stackunderflow.local")
PI_REMOTE = f"{PI_USER}@{PI_HOST}"
PI_AUDIO_DEVICE = os.getenv("PI_AUDIO_DEVICE", "plughw:2,0")
GPIO_PIN = int(os.getenv("GPIO_PIN", "17"))

# How many consecutive distress frames before alarm triggers
DISTRESS_THRESHOLD = 5

SITE_COLORS = {
    "sternum_cpr": (0, 0, 255),
    "outer_thigh_epipen": (0, 165, 255),
    "neck_pulse": (255, 0, 255),
    "chest_center_aed": (0, 255, 255),
}


# ── Alarm audio (pre-generated alarm.mp3 in project root) ──────────

ALARM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarm.mp3")


# ── SSH camera stream ────────────────────────────────────────────────

def _start_ssh_stream(width=640, height=480, fps=15):
    cmd = (
        f"rpicam-vid -t 0 --width {width} --height {height} "
        f"--framerate {fps} --codec mjpeg --inline -n -o -"
    )
    proc = subprocess.Popen(
        ["ssh", PI_REMOTE, cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    print(f"[stream] Pi camera streaming via SSH ({width}x{height} @ {fps}fps)")
    return proc


def _read_mjpeg_frames(pipe):
    buf = b""
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                buf = b""
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                buf = buf[start:]
                break
            frame_data = buf[start : end + 2]
            buf = buf[end + 2 :]
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is not None:
                yield frame


# ── Alarm controller ─────────────────────────────────────────────────

class AlarmController:
    """Plays emergency alarm on Pi via persistent paramiko connection."""

    def __init__(self):
        self._client = None
        self._playing = False
        self._lock = threading.Lock()

    def _connect(self):
        """Open a persistent SSH connection for alarm control."""
        if self._client and self._client.get_transport() and self._client.get_transport().is_active():
            return
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._client.connect(
            hostname=PI_HOST, username=PI_USER, password=PI_PASSWORD,
            look_for_keys=True, allow_agent=True, timeout=10,
        )
        print("[alarm] Paramiko connection established")

    def upload(self):
        """Upload pre-generated alarm MP3 to Pi."""
        self._connect()
        sftp = self._client.open_sftp()
        sftp.put(ALARM_FILE, "/tmp/alarm.mp3")
        sftp.close()
        print("[alarm] Uploaded alarm sound to Pi")

    def start(self):
        with self._lock:
            if self._playing:
                return
            try:
                self._connect()
                cmd = "while true; do mpg123 -o pulse /tmp/alarm.mp3 2>/dev/null; done"
                self._client.exec_command(cmd)
                self._playing = True
                print("[alarm] ALARM STARTED (looping on Pi)")
            except Exception as e:
                print(f"[alarm] Failed to start: {e}")

    def stop(self):
        with self._lock:
            if not self._playing:
                return
            try:
                self._connect()
                self._client.exec_command("pkill -9 -f 'alarm.mp3' 2>/dev/null; pkill -9 mpg123 2>/dev/null; exit 0")
                time.sleep(0.2)
            except Exception:
                pass
            self._playing = False
            print("[alarm] ALARM STOPPED")

    def is_playing(self):
        with self._lock:
            return self._playing

    def close(self):
        """Close the persistent connection."""
        try:
            if self._playing:
                self.stop()
            if self._client:
                self._client.close()
        except Exception:
            pass


# ── GPIO button listener ────────────────────────────────────────────

PI_PASSWORD = os.getenv("PI_PASSWORD", "tanishgoat")

class ButtonListener:
    """Listens for button press on Pi GPIO via paramiko (shares one TCP connection)."""

    def __init__(self, pin=17):
        self.pin = pin
        self.pressed = threading.Event()
        self._running = False
        self._client = None

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"[button] Starting button listener on GPIO {self.pin}")

    def _connect(self):
        """Open a dedicated paramiko SSH connection for button listening."""
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._client.connect(
            hostname=PI_HOST,
            username=PI_USER,
            password=PI_PASSWORD,
            look_for_keys=True,
            allow_agent=True,
            timeout=10,
        )
        print("[button] Paramiko connection established")

    def _upload_script(self):
        """Upload button script via SFTP."""
        script = (
            "#!/usr/bin/env python3\n"
            "import time\n"
            "from gpiozero import Button\n"
            f"b = Button({self.pin}, pull_up=True, bounce_time=0.05)\n"
            "print('READY', flush=True)\n"
            "while True:\n"
            "    b.wait_for_press()\n"
            "    print('PRESSED', flush=True)\n"
            "    time.sleep(0.3)\n"
        )
        sftp = self._client.open_sftp()
        with sftp.file("/tmp/button_listener.py", "w") as f:
            f.write(script)
        sftp.close()
        print("[button] Script uploaded to Pi")

    def _cleanup_gpio(self):
        """Kill any old button listener processes on Pi to free GPIO."""
        try:
            self._client.exec_command(
                "pkill -f button_listener 2>/dev/null; "
                "pkill -f test_button 2>/dev/null; "
                "pkill -f wait_for_press 2>/dev/null; "
                "sleep 0.5"
            )[1].read()  # wait for completion
            print("[button] Cleaned up old GPIO processes")
        except Exception:
            pass

    def _loop(self):
        while self._running:
            try:
                self._connect()
                self._cleanup_gpio()
                self._upload_script()

                # Run script and read output line by line
                stdin, stdout, stderr = self._client.exec_command(
                    "python3 /tmp/button_listener.py"
                )
                print("[button] Listener running on Pi")

                for line in stdout:
                    line = line.strip()
                    if not self._running:
                        break
                    if line == "READY":
                        print("[button] Listening for presses...")
                    elif "PRESSED" in line:
                        print("[button] BUTTON PRESSED!")
                        self.pressed.set()
                        while self.pressed.is_set() and self._running:
                            time.sleep(0.1)

                err = stderr.read().decode()
                if err:
                    print(f"[button] stderr: {err.strip()}")

            except Exception as e:
                print(f"[button] Error: {e}")
                time.sleep(3)

    def was_pressed(self):
        return self.pressed.is_set()

    def consume(self):
        self.pressed.clear()

    def stop(self):
        self._running = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass


# ── YOLO Pose processing ────────────────────────────────────────────

def _run_distress_checks(kps_numpy, img_h):
    detections = []

    nose = _kp(kps_numpy, NOSE)
    l_shoulder = _kp(kps_numpy, L_SHOULDER)
    r_shoulder = _kp(kps_numpy, R_SHOULDER)
    l_wrist = _kp(kps_numpy, L_WRIST)
    r_wrist = _kp(kps_numpy, R_WRIST)
    l_hip = _kp(kps_numpy, L_HIP)
    r_hip = _kp(kps_numpy, R_HIP)
    l_ankle = _kp(kps_numpy, L_ANKLE)
    r_ankle = _kp(kps_numpy, R_ANKLE)

    mid_shoulder = _mid(l_shoulder, r_shoulder)
    mid_hip = _mid(l_hip, r_hip)

    # CHECK 1: Fallen — head at or below hip level
    if nose and mid_hip:
        if nose[1] - mid_hip[1] > 20:
            detections.append("head below hips — possible fall")

    # CHECK 2a: Torso is horizontal
    if mid_shoulder and mid_hip:
        torso_dy = abs(mid_shoulder[1] - mid_hip[1])
        torso_dx = abs(mid_shoulder[0] - mid_hip[0])
        if torso_dx > torso_dy * 1.5 and torso_dy < 100:
            detections.append("torso horizontal — lying down")

    # CHECK 2b: Body spread wider than tall
    all_pts = [p for p in [nose, l_shoulder, r_shoulder, l_hip, r_hip,
                           l_ankle, r_ankle, l_wrist, r_wrist] if p]
    if len(all_pts) >= 4:
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        bw = max(all_x) - min(all_x)
        bh = max(all_y) - min(all_y)
        if bh > 0 and bw / bh > 1.8:
            detections.append("body spread horizontal — on the ground")

    # CHECK 3: Hands on/above head or covering face
    if nose and mid_shoulder:
        head_y = nose[1]
        shoulder_y = mid_shoulder[1]
        margin = abs(shoulder_y - head_y) * 0.5
        face_zone_bottom = shoulder_y + margin
        hands_above = 0
        hands_face = 0
        for wrist in [l_wrist, r_wrist]:
            if wrist is None:
                continue
            if wrist[1] < head_y:
                hands_above += 1
            elif wrist[1] < face_zone_bottom:
                hands_face += 1
        if hands_above >= 1:
            detections.append("hands above head — distress")
        elif hands_face + hands_above >= 2:
            detections.append("hands covering face — distress")
        elif hands_face >= 1:
            detections.append("hand near face — possible distress")

    # CHECK 4: Crouched / curled
    if mid_shoulder and mid_hip and nose:
        torso = abs(mid_shoulder[1] - mid_hip[1])
        head_to_hip = abs(nose[1] - mid_hip[1])
        if torso < 40 and head_to_hip < 60:
            detections.append("crouched/curled posture")

    # CHECK 5: Body low in frame
    visible = [p for p in [nose, l_shoulder, r_shoulder, l_hip, r_hip] if p]
    if len(visible) >= 3:
        avg_y = sum(p[1] for p in visible) / len(visible)
        if avg_y > img_h * 0.75:
            detections.append("body very low — possible collapse")

    return detections


def _process_frame(model, frame):
    start = time.time()
    results = model(frame, verbose=False)
    inference_ms = (time.time() - start) * 1000

    pose_result = {
        "is_distressed": False,
        "reason": "No person detected",
        "detections": [],
        "anatomical_sites": {},
        "num_persons": 0,
        "inference_ms": inference_ms,
    }

    if not results or len(results[0].keypoints) == 0:
        return results, pose_result

    kps_data = results[0].keypoints.data
    pose_result["num_persons"] = len(kps_data)
    img_h = frame.shape[0]
    all_detections = []
    all_sites = {}

    for i, kps in enumerate(kps_data):
        kps_np = kps.cpu().numpy()
        label = f"Person {i+1}"
        for d in _run_distress_checks(kps_np, img_h):
            all_detections.append(f"{label}: {d}")
        for name, coords in _get_anatomical_sites(kps_np).items():
            all_sites[f"{label}_{name}"] = coords

    pose_result["detections"] = all_detections
    pose_result["anatomical_sites"] = all_sites
    pose_result["is_distressed"] = len(all_detections) > 0
    pose_result["reason"] = (
        "; ".join(all_detections) if all_detections
        else "No distress posture detected"
    )
    return results, pose_result


# ── Frame annotation ─────────────────────────────────────────────────

def _annotate_frame(frame, results, pose_result, state="MONITORING"):
    annotated = results[0].plot(img=frame.copy())

    for name, (x, y) in pose_result.get("anatomical_sites", {}).items():
        ix, iy = int(x), int(y)
        color = (0, 255, 0)
        short_label = name
        for site_key, c in SITE_COLORS.items():
            if site_key in name:
                color = c
                short_label = site_key.replace("_", " ").upper()
                break
        cv2.drawMarker(annotated, (ix, iy), color, cv2.MARKER_CROSS, 20, 2)
        cv2.circle(annotated, (ix, iy), 12, color, 2)
        cv2.putText(annotated, short_label, (ix + 15, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    h, w = annotated.shape[:2]

    if state == "ALARM":
        flash = int(time.time() * 4) % 2 == 0
        bg = (0, 0, 255) if flash else (0, 0, 180)
        cv2.rectangle(annotated, (0, 0), (w, 35), bg, -1)
        cv2.putText(annotated, f"EMERGENCY! {pose_result['reason']}", (5, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    elif state == "PIPELINE":
        cv2.rectangle(annotated, (0, 0), (w, 35), (200, 100, 0), -1)
        cv2.putText(annotated, "ANALYZING... Gemini + TTS in progress", (5, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    elif pose_result.get("is_distressed"):
        cv2.rectangle(annotated, (0, 0), (w, 35), (0, 0, 200), -1)
        cv2.putText(annotated, f"DISTRESS: {pose_result['reason']}", (5, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(annotated, (0, 0), (w, 35), (0, 150, 0), -1)
        cv2.putText(annotated, "MONITORING — No distress", (5, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    ms = pose_result.get("inference_ms", 0)
    fps_val = 1000 / ms if ms > 0 else 0
    cv2.putText(annotated, f"{ms:.0f}ms ({fps_val:.0f} FPS)", (w - 160, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(annotated, f"[{state}]  Q: quit", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


# ── Gemini + TTS pipeline ───────────────────────────────────────────

def _run_pipeline(frame, pose_result, out_dir="captures"):
    """Capture frame -> Gemini -> ElevenLabs TTS -> play on Pi earphone."""
    from vision import analyze_image
    from speech import text_to_speech

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = str(out_path / f"{stem}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"\n[pipeline] Frame saved -> {image_path}")

    sites_str = ", ".join(
        f"{name}: ({x:.0f},{y:.0f})"
        for name, (x, y) in pose_result["anatomical_sites"].items()
    ) or "none"

    prompt = (
        "You are an emergency medical assistant guiding a bystander through a crisis. "
        f"YOLO Pose detected: {pose_result['reason']}. "
        f"Anatomical landmarks: {sites_str}. "
        "In 2-3 calm, clear sentences: describe what you see, assess the situation, "
        "and give the single most important first-aid instruction RIGHT NOW."
    )

    print("[pipeline] Calling Gemini...")
    description = analyze_image(image_path, prompt=prompt)
    print(f"[pipeline] Gemini: {description}")
    (out_path / f"{stem}.txt").write_text(description, encoding="utf-8")

    print("[pipeline] Converting to speech...")
    audio_out = str(out_path / f"{stem}.mp3")
    audio_path = text_to_speech(description, output_path=audio_out)

    print("[pipeline] Playing guidance on Pi earphone...")
    remote_mp3 = "/tmp/guidance.mp3"
    try:
        # Upload via paramiko SFTP
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=PI_HOST, username=PI_USER, password=PI_PASSWORD,
            look_for_keys=True, allow_agent=True, timeout=10,
        )
        sftp = client.open_sftp()
        sftp.put(audio_path, remote_mp3)
        sftp.close()
        print("[pipeline] Audio uploaded to Pi")

        # Kill any leftover alarm audio (bash loop + mpg123) before playing guidance
        client.exec_command("pkill -9 -f 'alarm.mp3' 2>/dev/null; pkill -9 mpg123 2>/dev/null")
        time.sleep(0.3)

        # Play MP3 via PulseAudio (routes to Bluetooth)
        stdin, stdout, stderr = client.exec_command(
            f"mpg123 -o pulse {remote_mp3}"
        )
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode()
            print(f"[pipeline] Play error (exit {exit_code}): {err}")
        else:
            print("[pipeline] Guidance played on Pi earphone!")
        client.close()
    except Exception as e:
        print(f"[pipeline] Audio playback failed: {e}")
    print("[pipeline] Done.\n")


# ── Main loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Guardian Eye — Live YOLO Pose + Alarm + Gemini"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out-dir", default="captures")
    parser.add_argument("--pin", type=int, default=GPIO_PIN)
    args = parser.parse_args()

    print("=" * 55)
    print("  GUARDIAN EYE — Live Distress Detection")
    print("=" * 55)

    # Load YOLO
    print("\n[init] Loading YOLO Pose model...")
    model = _load_model()
    print("[init] Model ready")

    # Upload alarm to Pi
    alarm = AlarmController()
    print("[init] Uploading alarm sound to Pi...")
    alarm.upload()

    # Start button listener
    button = ButtonListener(pin=args.pin)
    button.start()

    # Start camera stream
    stream_proc = _start_ssh_stream(args.width, args.height, args.fps)
    print("[init] Waiting for first frame...")
    time.sleep(2)

    print("\n[live] Monitoring for distress...")
    print("[live] Alarm triggers on distress. Press Pi button for Gemini guidance.\n")

    state = "MONITORING"
    pipeline_thread = None
    distress_streak = 0
    frame_count = 0
    last_button_time = 0  # cooldown: ignore presses within 5s

    try:
        for frame in _read_mjpeg_frames(stream_proc.stdout):
            frame_count += 1
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            results, pose_result = _process_frame(model, frame)

            # ── Button press: works in ANY state (except PIPELINE), 5s cooldown ──
            if button.was_pressed() and state != "PIPELINE":
                if time.time() - last_button_time < 5:
                    button.consume()  # eat the press but ignore it
                    print("[button] Ignored — cooldown (5s)")
                else:
                    button.consume()
                    last_button_time = time.time()
                    if state == "ALARM":
                        alarm.stop()
                    state = "PIPELINE"
                    print("[state] -> PIPELINE: Button pressed! Running Gemini + TTS...")
                    pipeline_thread = threading.Thread(
                        target=_run_pipeline,
                        args=(frame.copy(), pose_result, args.out_dir),
                        daemon=True,
                    )
                    pipeline_thread.start()

            # ── MONITORING: watch for distress ──
            if state == "MONITORING":
                if pose_result["is_distressed"]:
                    distress_streak += 1
                    if distress_streak >= DISTRESS_THRESHOLD:
                        state = "ALARM"
                        alarm.start()
                        print(f"[state] -> ALARM: {pose_result['reason']}")
                else:
                    distress_streak = 0

            # ── ALARM: stop if distress clears ──
            elif state == "ALARM":
                if pose_result["is_distressed"]:
                    distress_streak = DISTRESS_THRESHOLD  # keep it maxed
                else:
                    distress_streak -= 1
                    if distress_streak <= 0:
                        alarm.stop()
                        state = "MONITORING"
                        distress_streak = 0
                        print("[state] -> MONITORING: Distress cleared, alarm stopped\n")

            # ── PIPELINE: wait for completion ──
            elif state == "PIPELINE":
                if pipeline_thread and not pipeline_thread.is_alive():
                    pipeline_thread = None
                    state = "MONITORING"
                    distress_streak = 0
                    print("[state] -> MONITORING: Resuming\n")

            # Display
            annotated = _annotate_frame(frame, results, pose_result, state)
            cv2.imshow("Guardian Eye", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[live] Interrupted")
    finally:
        try:
            alarm.close()
        except Exception:
            pass
        button.stop()
        cv2.destroyAllWindows()
        stream_proc.terminate()
        stream_proc.wait()
        print(f"[live] Stopped after {frame_count} frames")


if __name__ == "__main__":
    main()
