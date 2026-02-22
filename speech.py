import os
import tempfile
import time
import requests
from dotenv import load_dotenv

from pi_connection import run_command, upload_file

load_dotenv()

ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip() or None
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM").strip()
TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
TTS_STREAM_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"


def text_to_speech(text, output_path=None):
    """Convert text to speech using ElevenLabs streaming API (faster first byte)."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Missing ELEVENLABS_API_KEY. Set it in your .env file.")
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "speech_output.mp3")

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    start = time.time()
    response = requests.post(
        TTS_STREAM_URL, json=payload, headers=headers, timeout=30, stream=True
    )
    response.raise_for_status()

    total_bytes = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    elapsed = time.time() - start
    print(f"[speech] Audio saved -> {output_path} ({total_bytes} bytes, {elapsed:.1f}s)")
    return output_path


def play_audio_on_pi(audio_path):
    """Upload audio to Pi and play on USB speaker over persistent connection."""
    remote_mp3 = "/tmp/speech_output.mp3"

    # Upload over persistent SFTP (no new connection)
    upload_file(audio_path, remote_mp3)
    print(f"[speech] Audio uploaded to Pi")

    # Play MP3 via PulseAudio (routes to Bluetooth)
    run_command(f"mpg123 -o pulse {remote_mp3}")
    print("[speech] Audio played on Pi USB speaker")


def speak(text):
    """Full pipeline: convert text to speech on laptop, play on Pi."""
    audio_path = text_to_speech(text)
    play_audio_on_pi(audio_path)
    return audio_path


if __name__ == "__main__":
    from pi_connection import connect, disconnect
    connect()
    speak("Hello! I am your Raspberry Pi assistant. I can see and speak!")
    disconnect()
