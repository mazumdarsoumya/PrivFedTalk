# Frame extraction helper (optional).
import os, cv2
from privfedtalk.utils.io import ensure_dir

def extract_frames(video_path: str, out_dir: str):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), frame)
        i += 1
    cap.release()
    return i
