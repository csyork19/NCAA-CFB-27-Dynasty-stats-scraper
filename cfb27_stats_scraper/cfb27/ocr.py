"""Thin wrapper around RapidOCR so the rest of the code deals in simple TextBox objects."""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

Region = Tuple[int, int, int, int]  # x0, y0, x1, y1 (pixels)


@dataclass
class TextBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    def __repr__(self) -> str:
        return f"<{self.text!r} @({self.cx:.0f},{self.cy:.0f})>"


class OCR:
    """Lazy RapidOCR engine. Works with `rapidocr` 2.x/3.x and the older `rapidocr_onnxruntime`."""

    def __init__(self, min_score: float = 0.3, threads: int = 0):
        self.min_score = min_score
        self.threads = threads
        self._engine = None
        self._new_api = True

    def _load(self):
        if self._engine is not None:
            return
        try:
            from rapidocr import RapidOCR  # type: ignore

            params = {"Global.log_level": "error", "Global.use_cls": False, "Global.text_score": self.min_score}
            if self.threads:
                params["EngineConfig.onnxruntime.intra_op_num_threads"] = self.threads
            self._engine = RapidOCR(params=params)
            self._new_api = True
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            kw = {"use_angle_cls": False, "text_score": self.min_score}
            if self.threads:
                kw["intra_op_num_threads"] = self.threads
            self._engine = RapidOCR(**kw)
            self._new_api = False

    def read(self, frame: np.ndarray, region: Optional[Region] = None, gray: bool = False,
             upscale: float = 1.0) -> List[TextBox]:
        self._load()
        x0, y0 = 0, 0
        img = frame
        if region is not None:
            x0, y0, x1, y1 = (int(v) for v in region)
            img = frame[max(0, y0):y1, max(0, x0):x1]
            x0, y0 = max(0, x0), max(0, y0)
        if img.size == 0:
            return []
        if gray:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if upscale != 1.0:
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        pad = 8
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        boxes: Sequence = []
        txts: Sequence = []
        scores: Sequence = []
        if self._new_api:
            res = self._engine(img)
            if res is not None and getattr(res, "boxes", None) is not None:
                boxes, txts, scores = res.boxes, res.txts, res.scores
        else:
            res, _ = self._engine(img)
            if res:
                boxes = [r[0] for r in res]
                txts = [r[1] for r in res]
                scores = [float(r[2]) for r in res]

        out: List[TextBox] = []
        for b, t, s in zip(boxes, txts, scores):
            s = float(s)
            if s < self.min_score or not str(t).strip():
                continue
            pts = np.asarray(b, dtype=float)
            bx0, by0 = (pts.min(axis=0) - pad) / upscale
            bx1, by1 = (pts.max(axis=0) - pad) / upscale
            txt = unicodedata.normalize("NFKC", str(t)).strip()
            out.append(TextBox(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0, txt, s))
        return out

    def recognize(self, crop: np.ndarray) -> Tuple[str, float]:
        """Recognition only (no text detection) on a small crop. ~10 ms."""
        self._load()
        if self._new_api:
            res = self._engine(crop, use_det=False, use_cls=False, use_rec=True)
            txts = getattr(res, "txts", None) or [""]
            scores = getattr(res, "scores", None) or [0.0]
            return unicodedata.normalize("NFKC", str(txts[0])), float(scores[0])
        r = self._engine.text_recognizer([crop])[0][0]
        return unicodedata.normalize("NFKC", str(r[0])), float(r[1])

    def read_short(self, cell: np.ndarray) -> Tuple[str, float]:
        """Read a tiny cell (one or two characters, which full OCR tends to miss).
        The ink is cropped tightly and repeated three times, which recognisers handle far
        better than a lone character, then the repeated token is recovered."""
        g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.int16)
        bg = float(np.median(g))
        ys, xs = np.where(np.abs(g - bg) > 60)
        if len(xs) < 6:
            return "", 0.0  # blank cell
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h = y1 - y0 + 1
        p = max(3, int(0.35 * h))
        c = cell[max(0, y0 - p):y1 + p + 1, max(0, x0 - p):x1 + p + 1]
        fill = np.median(c.reshape(-1, 3), axis=0).astype(np.uint8)
        gap = np.full((c.shape[0], max(6, c.shape[0] // 2), 3), fill, np.uint8)
        tiled = np.hstack([c, gap, c, gap, c])
        tiled = cv2.copyMakeBorder(tiled, 0, 0, 8, 8, cv2.BORDER_REPLICATE)
        txt, score = self.recognize(tiled)
        return _untile(txt.replace(" ", ""), 3), score


def _untile(s: str, n: int) -> str:
    if not s:
        return ""
    best, best_d = None, 99
    for L in range(1, len(s) + 1):
        if len(s) % L == 0 and s == s[:L] * (len(s) // L):
            d = abs(len(s) // L - n)
            if d < best_d:
                best, best_d = s[:L], d
    if best is not None and best_d <= 1:
        return best
    if len(s) % n == 0:
        parts = [s[i * len(s) // n:(i + 1) * len(s) // n] for i in range(n)]
        top = max(set(parts), key=parts.count)
        if parts.count(top) >= 2:
            return top
    return ""
