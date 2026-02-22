"""
Single persistent SSH connection to the Raspberry Pi.

Opens ONE connection at startup, reuses it for all commands and file transfers.
Eliminates ~3s SSH handshake overhead per call.
"""

import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

PI_USER = os.getenv("PI_USER", "stackunderflow")
PI_HOST = os.getenv("PI_HOST", "stackunderflow.local")
PI_PASSWORD = os.getenv("PI_PASSWORD", "tanishgoat")

# USB speaker on Pi (card 2, device 0)
PI_AUDIO_DEVICE = "plughw:2,0"

_client = None
_sftp = None


def connect():
    """Open a persistent SSH + SFTP connection to the Pi."""
    global _client, _sftp
    if _client and _client.get_transport() and _client.get_transport().is_active():
        return  # Already connected

    print(f"[ssh] Connecting to {PI_USER}@{PI_HOST}...")
    _client = paramiko.SSHClient()
    _client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    _client.connect(
        hostname=PI_HOST,
        username=PI_USER,
        password=PI_PASSWORD,
        look_for_keys=True,
        allow_agent=True,
        timeout=10,
    )
    _sftp = _client.open_sftp()
    print("[ssh] Connected (persistent connection)")


def run_command(cmd):
    """Run a command on the Pi over the persistent connection."""
    if not _client:
        connect()
    stdin, stdout, stderr = _client.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    if exit_code != 0:
        raise RuntimeError(f"Remote command failed (exit {exit_code}): {err.strip()}")
    return out.strip()


def upload_file(local_path, remote_path):
    """Copy a file from laptop to Pi over the persistent connection."""
    if not _sftp:
        connect()
    _sftp.put(local_path, remote_path)


def download_file(remote_path, local_path):
    """Copy a file from Pi to laptop over the persistent connection."""
    if not _sftp:
        connect()
    _sftp.get(remote_path, local_path)


def disconnect():
    """Close the persistent connection."""
    global _client, _sftp
    if _sftp:
        _sftp.close()
        _sftp = None
    if _client:
        _client.close()
        _client = None
    print("[ssh] Disconnected")
