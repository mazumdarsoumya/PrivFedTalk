# Audio extraction helper (optional, requires ffmpeg).
import subprocess, os
from privfedtalk.utils.io import ensure_dir

def extract_audio(video_path: str, out_wav: str, sr: int = 16000):
    ensure_dir(os.path.dirname(out_wav))
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), out_wav]
    subprocess.check_call(cmd)
    return out_wav
