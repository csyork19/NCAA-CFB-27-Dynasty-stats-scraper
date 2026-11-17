"""Parallel OCR: the navigation loop only saves screenshots; worker processes read them.

Navigation never waits for OCR, so the run is limited by how fast the game's menus move,
not by how fast text can be read.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_OCR = None


def _init(threads: int) -> None:
    global _OCR
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, threads)))
    from .ocr import OCR

    _OCR = OCR(threads=threads)


def _work(path: str, meta: dict) -> dict:
    import cv2

    from . import vision as V

    img = cv2.imread(path)
    if img is None:
        return {"meta": meta, "table": None, "error": "unreadable image"}
    frame = V.to_ref(img)
    try:
        boxes = _OCR.read(frame)
        table = V.parse_table(boxes)
        V.fill_missing(frame, table, _OCR)
        return {"meta": meta, "table": table.to_dict()}
    except Exception as e:  # keep going; the merge step just skips this page
        return {"meta": meta, "table": None, "error": repr(e)}


def default_workers() -> int:
    n = os.cpu_count() or 4
    return max(1, (n - 2) // 2)  # leave cores for the game; each worker uses 2 threads


class OcrPool:
    def __init__(self, workers: int = 0, threads: int = 2):
        self.workers = workers or default_workers()
        self.exe = ProcessPoolExecutor(max_workers=self.workers, initializer=_init, initargs=(threads,))
        self.futures: List[Future] = []
        self.results: List[dict] = []
        self._lock = threading.Lock()
        log.info("OCR pool: %d worker processes", self.workers)

    def submit(self, path: str, meta: dict) -> None:
        fut = self.exe.submit(_work, path, meta)
        fut.add_done_callback(self._done)
        self.futures.append((meta.get("kind", ""), fut))

    def _done(self, fut: Future) -> None:
        try:
            res = fut.result()
        except Exception as e:
            log.warning("OCR job failed: %s", e)
            return
        with self._lock:
            self.results.append(res)

    def pending(self, prefix: str = "") -> int:
        return sum(1 for k, f in self.futures if k.startswith(prefix) and not f.done())

    def wait(self, prefix: str = "", timeout: Optional[float] = None) -> List[dict]:
        """Block until the jobs whose meta.kind starts with `prefix` are done."""
        import time

        t0 = time.time()
        while self.pending(prefix) and not (timeout and time.time() - t0 > timeout):
            time.sleep(0.2)
        return self.snapshot()

    def snapshot(self) -> List[dict]:
        with self._lock:
            return list(self.results)

    def close(self) -> None:
        self.exe.shutdown(wait=True)


class CaptureLog:
    """Writes screenshots + an index so the OCR/merge can be re-run later without the game."""

    def __init__(self, folder: Path, quality: int = 92):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = open(folder / "index.jsonl", "a", encoding="utf-8")
        self.quality = quality
        self.n = 0

    def save(self, frame, meta: dict) -> str:
        import cv2

        self.n += 1
        name = f"{self.n:05d}_{meta.get('kind', 'x').replace(':', '-')}_t{meta.get('team_idx', 0)}" \
               f"_v{meta.get('v', 0)}_h{meta.get('h', 0)}.jpg"
        path = str(self.folder / name)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        rec = dict(meta, file=name)
        self.index.write(json.dumps(rec) + "\n")
        self.index.flush()
        return path

    def close(self) -> None:
        self.index.close()


def load_index(folder: Path) -> List[Dict]:
    out = []
    with open(folder / "index.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
