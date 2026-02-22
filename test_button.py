"""Quick test: does the Pi button work over SSH?"""
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

PI_USER = os.getenv("PI_USER", "stackunderflow")
PI_HOST = os.getenv("PI_HOST", "stackunderflow.local")
PI_REMOTE = f"{PI_USER}@{PI_HOST}"
PIN = 17

# Upload script
script = (
    "import time\n"
    "from gpiozero import Button\n"
    f"b = Button({PIN}, pull_up=True, bounce_time=0.05)\n"
    "print('READY - press the button!', flush=True)\n"
    "while True:\n"
    "    b.wait_for_press()\n"
    "    print('PRESSED!', flush=True)\n"
    "    time.sleep(0.3)\n"
)

print("[1] Uploading script to Pi...")
proc = subprocess.Popen(
    ["ssh", PI_REMOTE, "cat > /tmp/test_button.py"],
    stdin=subprocess.PIPE,
)
proc.communicate(input=script.encode(), timeout=10)
print("[2] Script uploaded. Running...\n")

# Run and print output live
proc = subprocess.Popen(
    ["ssh", PI_REMOTE, "python3 /tmp/test_button.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

print("Waiting for output (Ctrl+C to quit)...\n")
try:
    for line in proc.stdout:
        print(f"  >>> {line.strip()}")
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    err = proc.stderr.read()
    if err:
        print(f"\nSTDERR: {err.strip()}")
    proc.terminate()
