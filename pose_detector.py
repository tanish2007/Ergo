"""
YOLO Pose-based distress detection with anatomical landmark mapping.

Runs YOLOv8n-pose locally to detect body posture and determine if a person
is in distress (fallen, collapsed, hands on head, etc.)

Also maps key anatomical sites for medical guidance:
  - Sternum (CPR location) — midpoint between shoulders
  - Outer thigh (EpiPen) — midpoint of hip-to-knee on outer side
  - Neck (pulse check) — between ear and shoulder

COCO Keypoint indices:
  0: nose        1: left_eye     2: right_eye
  3: left_ear    4: right_ear    5: left_shoulder
  6: right_shoulder  7: left_elbow   8: right_elbow
  9: left_wrist  10: right_wrist  11: left_hip
  12: right_hip  13: left_knee    14: right_knee
  15: left_ankle  16: right_ankle
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load model once at import time
_model = None

# Keypoint indices
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Confidence threshold for a keypoint to be considered visible
KP_CONF = 0.3


def _load_model():
    global _model
    if _model is None:
        print("[pose] Loading YOLOv8n-pose model...")
        _model = YOLO("yolov8n-pose.pt")
        print("[pose] Model loaded")
    return _model


def _kp(keypoints, idx):
    """Return (x, y, conf) for a keypoint, or None if below confidence."""
    x, y, conf = keypoints[idx]
    if conf < KP_CONF:
        return None
    return (float(x), float(y), float(conf))


def _mid(a, b):
    """Midpoint of two (x,y,conf) keypoints."""
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _get_anatomical_sites(kps):
    """
    Map anatomical sites from pose keypoints for medical guidance.
    Returns dict of site_name -> (x, y) or None.
    """
    l_shoulder = _kp(kps, L_SHOULDER)
    r_shoulder = _kp(kps, R_SHOULDER)
    l_hip = _kp(kps, L_HIP)
    r_hip = _kp(kps, R_HIP)
    l_knee = _kp(kps, L_KNEE)
    r_knee = _kp(kps, R_KNEE)
    l_ear = _kp(kps, L_EAR)
    r_ear = _kp(kps, R_EAR)

    sites = {}

    # Sternum — CPR target — midpoint between shoulders
    sternum = _mid(l_shoulder, r_shoulder)
    if sternum:
        sites["sternum_cpr"] = sternum

    # Outer thigh — EpiPen injection site — outer mid-thigh
    if l_hip and l_knee:
        sites["left_outer_thigh_epipen"] = _mid(l_hip, l_knee)
    if r_hip and r_knee:
        sites["right_outer_thigh_epipen"] = _mid(r_hip, r_knee)

    # Neck / carotid — pulse check — between ear and shoulder
    if l_ear and l_shoulder:
        sites["left_neck_pulse"] = _mid(l_ear, l_shoulder)
    if r_ear and r_shoulder:
        sites["right_neck_pulse"] = _mid(r_ear, r_shoulder)

    # Center of chest — AED pad placement
    if sternum and l_hip and r_hip:
        mid_hip = _mid(l_hip, r_hip)
        sites["chest_center_aed"] = _mid(
            (sternum[0], sternum[1], 1), (mid_hip[0], mid_hip[1], 1)
        )

    return sites


def _draw_annotated_image(results, pose_result, output_path):
    """
    Draw YOLO skeleton + anatomical site markers + distress labels
    and save to output_path.
    """
    # Start with YOLO's own annotated frame (skeleton + bbox)
    annotated = results[0].plot()

    # Color scheme for anatomical sites
    site_colors = {
        "sternum_cpr": (0, 0, 255),          # Red — CPR
        "outer_thigh_epipen": (0, 165, 255),  # Orange — EpiPen
        "neck_pulse": (255, 0, 255),          # Magenta — pulse
        "chest_center_aed": (0, 255, 255),    # Yellow — AED
    }

    # Draw anatomical sites as labeled circles
    for name, (x, y) in pose_result["anatomical_sites"].items():
        ix, iy = int(x), int(y)

        # Pick color based on site type
        color = (0, 255, 0)  # default green
        short_label = name
        for site_key, c in site_colors.items():
            if site_key in name:
                color = c
                short_label = site_key.upper().replace("_", " ")
                break

        # Draw crosshair marker
        cv2.drawMarker(annotated, (ix, iy), color, cv2.MARKER_CROSS, 20, 2)
        cv2.circle(annotated, (ix, iy), 12, color, 2)

        # Label
        cv2.putText(
            annotated, short_label, (ix + 15, iy - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    # Draw distress status banner at top
    h, w = annotated.shape[:2]
    if pose_result["is_distressed"]:
        # Red banner
        cv2.rectangle(annotated, (0, 0), (w, 30), (0, 0, 200), -1)
        cv2.putText(
            annotated, f"DISTRESS: {pose_result['reason']}", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    else:
        # Green banner
        cv2.rectangle(annotated, (0, 0), (w, 30), (0, 150, 0), -1)
        cv2.putText(
            annotated, "NO DISTRESS DETECTED", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Inference time in bottom-right
    ms_text = f"{pose_result['inference_ms']:.0f}ms"
    cv2.putText(
        annotated, ms_text, (w - 70, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
    )

    cv2.imwrite(output_path, annotated)
    print(f"[pose] Annotated image saved -> {output_path}")


def detect_distress(image_path, annotated_path=None):
    """
    Run YOLO Pose on an image and check for distress postures.

    Args:
        image_path: path to input image
        annotated_path: if provided, save annotated image with skeleton + sites

    Returns:
        dict with keys:
          - is_distressed (bool)
          - reason (str)
          - detections (list of str)
          - anatomical_sites (dict of site_name -> (x,y))
          - num_persons (int)
          - inference_ms (float)
    """
    model = _load_model()

    start = time.time()
    results = model(image_path, verbose=False)
    inference_ms = (time.time() - start) * 1000
    print(f"[pose] YOLO inference: {inference_ms:.0f}ms")

    result = {
        "is_distressed": False,
        "reason": "No person detected",
        "detections": [],
        "anatomical_sites": {},
        "num_persons": 0,
        "inference_ms": inference_ms,
    }

    if not results or len(results[0].keypoints) == 0:
        return result

    kps_data = results[0].keypoints.data
    result["num_persons"] = len(kps_data)
    img_h = results[0].orig_shape[0]
    all_detections = []
    all_sites = {}

    for i, kps in enumerate(kps_data):
        kps = kps.cpu().numpy()
        label = f"Person {i+1}"

        # Extract keypoints
        nose = _kp(kps, NOSE)
        l_shoulder = _kp(kps, L_SHOULDER)
        r_shoulder = _kp(kps, R_SHOULDER)
        l_wrist = _kp(kps, L_WRIST)
        r_wrist = _kp(kps, R_WRIST)
        l_hip = _kp(kps, L_HIP)
        r_hip = _kp(kps, R_HIP)
        l_ankle = _kp(kps, L_ANKLE)
        r_ankle = _kp(kps, R_ANKLE)

        # Midpoints
        mid_shoulder = _mid(l_shoulder, r_shoulder)
        mid_hip = _mid(l_hip, r_hip)
        mid_ankle = _mid(l_ankle, r_ankle)

        # --- CHECK 1: Fallen — head at or below hip level ---
        if nose and mid_hip:
            if nose[1] - mid_hip[1] > 20:
                all_detections.append(f"{label}: head below hips — possible fall")

        # --- CHECK 2: Lying down — body is horizontal ---
        # Method A: torso line is horizontal (shoulders and hips at similar Y)
        if mid_shoulder and mid_hip:
            torso_dy = abs(mid_shoulder[1] - mid_hip[1])
            torso_dx = abs(mid_shoulder[0] - mid_hip[0])
            if torso_dx > torso_dy * 1.5 and torso_dy < 100:
                all_detections.append(f"{label}: torso horizontal — lying down")

        # Method B: overall body spread is wider than tall
        all_pts = [p for p in [nose, l_shoulder, r_shoulder, l_hip, r_hip,
                               l_ankle, r_ankle, l_wrist, r_wrist] if p]
        if len(all_pts) >= 4:
            all_x = [p[0] for p in all_pts]
            all_y = [p[1] for p in all_pts]
            body_width = max(all_x) - min(all_x)
            body_height = max(all_y) - min(all_y)
            if body_height > 0 and body_width / body_height > 1.8:
                all_detections.append(f"{label}: body spread horizontal — on the ground")

        # --- CHECK 3: Hands on/above head OR covering face ---
        # Face zone: from above the head to slightly below shoulders
        if nose and mid_shoulder:
            head_y = nose[1]
            shoulder_y = mid_shoulder[1]
            # Extend face zone 30% below shoulders to catch hands resting on face
            margin = abs(shoulder_y - head_y) * 0.5
            face_zone_bottom = shoulder_y + margin

            hands_above_head = 0
            hands_on_face = 0

            for wrist in [l_wrist, r_wrist]:
                if wrist is None:
                    continue
                if wrist[1] < head_y:
                    hands_above_head += 1
                elif wrist[1] < face_zone_bottom:
                    hands_on_face += 1

            if hands_above_head >= 1:
                all_detections.append(f"{label}: hands above head — distress posture")
            elif hands_on_face + hands_above_head >= 2:
                all_detections.append(f"{label}: hands covering face — distress posture")
            elif hands_on_face >= 1:
                all_detections.append(f"{label}: hand near face — possible distress")

        # --- CHECK 4: Crouched / curled up ---
        if mid_shoulder and mid_hip and nose:
            torso = abs(mid_shoulder[1] - mid_hip[1])
            head_to_hip = abs(nose[1] - mid_hip[1])
            if torso < 40 and head_to_hip < 60:
                all_detections.append(f"{label}: crouched/curled posture")

        # --- CHECK 5: Body low in frame — on the ground ---
        visible = [p for p in [nose, l_shoulder, r_shoulder, l_hip, r_hip] if p]
        if len(visible) >= 3:
            avg_y = sum(p[1] for p in visible) / len(visible)
            if avg_y > img_h * 0.75:
                all_detections.append(f"{label}: body very low in frame — possible collapse")

        # Get anatomical sites for this person
        person_sites = _get_anatomical_sites(kps)
        for site_name, coords in person_sites.items():
            all_sites[f"{label}_{site_name}"] = coords

    result["detections"] = all_detections
    result["anatomical_sites"] = all_sites
    result["is_distressed"] = len(all_detections) > 0

    if result["is_distressed"]:
        result["reason"] = "; ".join(all_detections)
        print(f"[pose] DISTRESS DETECTED: {result['reason']}")
    else:
        result["reason"] = "No distress posture detected"
        print("[pose] No distress posture detected")

    # Save annotated image if path provided
    if annotated_path:
        _draw_annotated_image(results, result, annotated_path)

    return result


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "download.jpg"
    r = detect_distress(img)
    print(f"\nPersons detected: {r['num_persons']}")
    print(f"Distressed: {r['is_distressed']}")
    print(f"Reason: {r['reason']}")
    print(f"Inference: {r['inference_ms']:.0f}ms")
    if r["anatomical_sites"]:
        print("\nAnatomical sites:")
        for name, (x, y) in r["anatomical_sites"].items():
            print(f"  {name}: ({x:.0f}, {y:.0f})")
