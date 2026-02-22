import os
import tempfile
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

from pi_connection import run_command, download_file

load_dotenv()

_GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)

REMOTE_CAPTURE_PATH = "/tmp/capture.jpg"


def capture_image_remote(output_path=None, width=640, height=480, timeout_ms=1000):
    """Capture an image on the Pi and download it over the persistent SSH connection."""
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "capture.jpg")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Capture on Pi (640x480 for speed — Gemini doesn't need high res)
    run_command(
        f"rpicam-still -o {REMOTE_CAPTURE_PATH} "
        f"--width {width} --height {height} -t {timeout_ms} -n"
    )
    print(f"[vision] Image captured on Pi")

    # Download over persistent SFTP (no new connection)
    download_file(REMOTE_CAPTURE_PATH, output_path)
    print(f"[vision] Image downloaded -> {output_path}")
    return output_path


def analyze_image(image_path, prompt="Describe what you see in this image in detail.", max_retries=3):
    """Send an image to Gemini and return the text response."""
    import time
    if not _GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in your .env file.")
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(image_path)

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, img])
            text = response.text
            print(f"[vision] Gemini response ({len(text)} chars)")
            return text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"[vision] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


if __name__ == "__main__":
    from pi_connection import connect, disconnect
    connect()
    path = capture_image_remote()
    result = analyze_image(path)
    print(result)
    disconnect()
