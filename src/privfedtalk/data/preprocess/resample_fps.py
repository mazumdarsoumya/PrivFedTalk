# FPS resampling helper (optional, requires ffmpeg).
import subprocess

def resample_fps(in_video: str, out_video: str, fps: int = 25):
    cmd = ["ffmpeg", "-y", "-i", in_video, "-r", str(fps), out_video]
    subprocess.check_call(cmd)
    return out_video
