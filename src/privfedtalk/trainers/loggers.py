import os, csv
from privfedtalk.utils.io import ensure_dir

class CSVLogger:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        try: self.f.close()
        except Exception: pass
