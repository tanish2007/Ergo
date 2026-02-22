#!/usr/bin/env python3
"""
Vision-to-Speech Pipeline (Hybrid: YOLO Pose + Gemini)
-------------------------------------------------------
1. Pi: button press -> camera capture
2. Laptop: YOLO Pose (~12ms) detects distress posture
3. If distress: Gemini provides detailed medical guidance
4. If no distress: instant "no distress" response
5. ElevenLabs TTS -> plays on Pi USB speaker

Single persistent SSH connection for all Pi communication.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pi_connection import connect, disconnect, run_command
from vision import capture_image_remote, analyze_image
from speech import text_to_speech, play_audio_on_pi
from pose_detector import detect_distress


def _timestamp_slug():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


DEFAULT_DISTRESS_PROMPT = (
    "You are an emergency medical assistant guiding a bystander through a crisis. "
    "A person in the image appears to be in distress. "
    "YOLO Pose detected: {pose_reason}. "
    "Anatomical landmarks identified: {sites}. "
    "In 2-3 calm, clear sentences: describe what you see, assess the situation, "
    "and give the single most important first-aid instruction the bystander should do RIGHT NOW. "
    "Be specific about body locations. Speak as if directly guiding someone who is panicking."
)

NO_DISTRESS_RESPONSE = "No distress detected. The scene appears safe."


def _build_gemini_prompt(pose_result):
    """Build a context-rich Gemini prompt using YOLO Pose data."""
    sites_str = ", ".join(
        f"{name}: ({x:.0f},{y:.0f})"
        for name, (x, y) in pose_result["anatomical_sites"].items()
    ) or "none identified"

    return DEFAULT_DISTRESS_PROMPT.format(
        pose_reason=pose_result["reason"],
        sites=sites_str,
    )


def run_pipeline(
    prompt=None,
    play=True,
    out_dir="captures",
    width=640,
    height=480,
    image_path_in=None,
    text_only=False,
):
    """Run the full hybrid pipeline once."""
    start = time.time()
    print("=" * 50)
    print("  HYBRID PIPELINE (YOLO Pose + Gemini)")
    print("=" * 50)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = _timestamp_slug()
    image_path_out = str(out_path / f"{stem}.jpg")

    # Step 1: Capture image on Pi, download to laptop
    if image_path_in:
        src = Path(image_path_in)
        if not src.exists():
            raise FileNotFoundError(f"Input image not found: {image_path_in}")
        print(f"\n[1/4] Using existing image: {src}")
        import shutil
        shutil.copy2(src, image_path_out)
        image_path = image_path_out
    else:
        print("\n[1/4] Capturing image on Pi...")
        image_path = capture_image_remote(
            output_path=image_path_out, width=width, height=height
        )

    # Step 2: YOLO Pose — fast local distress detection (~12ms)
    print("\n[2/4] Running YOLO Pose (local)...")
    pose_annotated = str(out_path / f"{stem}_pose.jpg")
    pose_result = detect_distress(image_path, annotated_path=pose_annotated)

    if not pose_result["is_distressed"]:
        # No distress — skip Gemini, respond immediately
        description = NO_DISTRESS_RESPONSE
        print(f"\n--- No distress ---\n{description}\n-------------------\n")
        (out_path / f"{stem}.txt").write_text(description, encoding="utf-8")

        if not text_only:
            print("[3/4] Skipping Gemini (no distress)")
            print("[4/4] Converting to speech and playing on Pi...")
            audio_out = str(out_path / f"{stem}.mp3")
            audio_path = text_to_speech(description, output_path=audio_out)
            if play:
                play_audio_on_pi(audio_path)

        elapsed = time.time() - start
        print(f"\nPipeline complete! ({elapsed:.1f}s) [FAST PATH — no Gemini]")
        return description, None

    # Step 3: Distress detected — call Gemini for detailed medical guidance
    gemini_prompt = prompt or _build_gemini_prompt(pose_result)
    print(f"\n[3/4] DISTRESS DETECTED — calling Gemini for medical guidance...")
    print(f"       Pose: {pose_result['reason']}")
    description = analyze_image(image_path, prompt=gemini_prompt)
    print(f"\n--- Gemini guidance ---\n{description}\n-----------------------\n")

    (out_path / f"{stem}.txt").write_text(description, encoding="utf-8")

    if text_only:
        print("[4/4] Text-only mode: skipping audio.")
        elapsed = time.time() - start
        print(f"\nPipeline complete! ({elapsed:.1f}s)")
        return description, None

    # Step 4: TTS on laptop, play on Pi
    print("[4/4] Converting to speech and playing on Pi...")
    audio_out = str(out_path / f"{stem}.mp3")
    audio_path = text_to_speech(description, output_path=audio_out)
    if play:
        play_audio_on_pi(audio_path)
    else:
        print(f"Audio saved to {audio_path} (playback skipped)")

    elapsed = time.time() - start
    print(f"\nPipeline complete! ({elapsed:.1f}s) [FULL PATH — Gemini + TTS]")
    return description, audio_path


def run_button_mode(
    pin,
    prompt=None,
    play=True,
    out_dir="captures",
    width=640,
    height=480,
):
    """
    Wait for button press on Pi, then run the hybrid pipeline.
    Loops until Ctrl+C.
    """
    print("=" * 50)
    print("  BUTTON MODE (YOLO Pose + Gemini)")
    print("=" * 50)
    print(f"Listening for button on Pi GPIO {pin}.")
    print(f"Saving outputs to: {os.path.abspath(out_dir)}")
    print("Press Ctrl+C to quit.\n")

    # Warm up YOLO model
    print("[init] Warming up YOLO Pose model...")
    from pose_detector import _load_model
    _load_model()
    print("[init] Ready!\n")

    while True:
        print(f"[button] Waiting for button press on GPIO {pin}...")
        run_command(
            f"python3 -c \""
            f"from gpiozero import Button; "
            f"b = Button({pin}, pull_up=True, bounce_time=0.05); "
            f"b.wait_for_press(); "
            f"print('PRESSED')"
            f"\""
        )
        print("[button] Button pressed!")

        try:
            run_pipeline(
                prompt=prompt, play=play, out_dir=out_dir,
                width=width, height=height,
            )
        except Exception as e:
            print(f"\nERROR during pipeline run: {e}\n")
        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Vision-to-Speech Pipeline (YOLO Pose + Gemini)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default=None,
        help="Custom prompt for Gemini (overrides auto-generated medical prompt)",
    )
    parser.add_argument(
        "--mode",
        choices=["button", "once"],
        default="button",
        help="Run once, or wait for GPIO button presses on Pi",
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=17,
        help="GPIO pin number for the pushbutton on Pi",
    )
    parser.add_argument(
        "--out-dir",
        default="captures",
        help="Directory on laptop to store outputs",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an existing image (skips camera capture)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only run detection + Gemini, skip audio",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Save audio without playing on Pi",
    )
    args = parser.parse_args()

    try:
        # Open ONE persistent SSH connection
        if not args.image:
            connect()

        if args.mode == "once":
            run_pipeline(
                prompt=args.prompt,
                play=not args.no_play,
                out_dir=args.out_dir,
                width=args.width,
                height=args.height,
                image_path_in=args.image,
                text_only=args.text_only,
            )
        else:
            if args.image or args.text_only:
                raise RuntimeError("--image/--text-only can only be used with --mode once")
            run_button_mode(
                pin=args.pin,
                prompt=args.prompt,
                play=not args.no_play,
                out_dir=args.out_dir,
                width=args.width,
                height=args.height,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    finally:
        disconnect()
