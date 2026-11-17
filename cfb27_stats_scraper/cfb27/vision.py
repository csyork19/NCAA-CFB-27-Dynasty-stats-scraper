"""Screen understanding that doesn't depend on fixed pixel positions.

Everything works on frames normalised to 1920x1080 (`to_ref`) and on OCR text boxes:
  * finding menu items by text and telling which one is highlighted,
  * finding the stats table on any screen (header row + rows of numbers) and reading it,
  * cheap pixel checks (no OCR) that tell whether a table scrolled, wrapped, or ended.
"""
from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ocr import TextBox

REF_W, REF_H = 1920, 1080


def to_ref(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]
    if (w, h) != (REF_W, REF_H):
        interp = cv2.INTER_AREA if w > REF_W else cv2.INTER_CUBIC
        frame = cv2.resize(frame, (REF_W, REF_H), interpolation=interp)
    return frame


# ------------------------------------------------------------------ text matching
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("&", "and"))


def ratio(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_eq(a: str, b: str, cutoff: float = 0.8) -> bool:
    return ratio(a, b) >= cutoff


def fuzzy_in(target: str, text: str, cutoff: float = 0.8) -> bool:
    """True if `target` appears (approximately) inside `text`."""
    t, x = norm(target), norm(text)
    if not t or not x:
        return False
    if t in x:
        return True
    if len(x) < len(t) - 1:
        return False
    n = len(t)
    for w in (n - 1, n, n + 1):
        for i in range(0, max(1, len(x) - w + 1)):
            if difflib.SequenceMatcher(None, t, x[i:i + w]).ratio() >= cutoff:
                return True
    return False


def find_text(boxes: Sequence[TextBox], target: str, cutoff: float = 0.78) -> Optional[TextBox]:
    """Best box whose text matches `target` (whole box, or target contained in a longer box)."""
    best, best_s = None, 0.0
    for b in boxes:
        s = ratio(b.text, target)
        if s < cutoff and fuzzy_in(target, b.text, 0.86) and len(norm(b.text)) <= len(norm(target)) + 12:
            s = cutoff
        if s > best_s:
            best, best_s = b, s
    return best if best_s >= cutoff else None


def count_markers(boxes: Sequence[TextBox], markers: Sequence[str], cutoff: float = 0.8) -> int:
    return sum(1 for m in markers if find_text(boxes, m, cutoff) is not None)


def best_choice(text: str, choices: Sequence[str], cutoff: float = 0.7) -> Optional[str]:
    if not choices or not norm(text):
        return None
    best = max(choices, key=lambda c: ratio(c, text))
    return best if ratio(best, text) >= cutoff else None


# ------------------------------------------------------------------ highlight detection
def _ring_color(frame: np.ndarray, b: TextBox) -> np.ndarray:
    m = int(max(6, 0.45 * b.h))
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(b.x0) - m), max(0, int(b.y0) - m)
    x1, y1 = min(W, int(b.x1) + m), min(H, int(b.y1) + m)
    patch = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if patch.size == 0:
        return np.zeros(3, np.float32)
    # background dominates the padded box; median ignores the glyph pixels
    return np.median(patch, axis=0)


def highlighted_box(frame: np.ndarray, boxes: Sequence[TextBox], candidates: Optional[Sequence[TextBox]] = None,
                    min_dist: float = 45.0) -> Optional[TextBox]:
    """The box sitting on a background that clearly differs from the rest (the menu cursor).
    `candidates` limits the answer to one menu (e.g. the target and its siblings)."""
    boxes = [b for b in boxes if len(norm(b.text)) >= 2]
    if len(boxes) < 2:
        return None
    rings = {id(b): _ring_color(frame, b) for b in boxes}
    med = np.median(np.array(list(rings.values())), axis=0)
    pool = [b for b in (candidates if candidates is not None else boxes) if id(b) in rings]
    if not pool:
        return None
    scored = sorted(((float(np.linalg.norm(rings[id(b)] - med)), b) for b in pool), key=lambda x: -x[0])
    top = scored[0][0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if top < min_dist or (second > 0.75 * top and second > min_dist):
        return None
    return scored[0][1]


def siblings(boxes: Sequence[TextBox], tgt: TextBox) -> List[TextBox]:
    """Other items of the same menu: similar text size, lined up in a row or a column."""
    out = []
    for b in boxes:
        if b is tgt or len(norm(b.text)) < 2:
            continue
        if not (0.75 < b.h / max(1.0, tgt.h) < 1.33):
            continue
        same_row = abs(b.cy - tgt.cy) < 0.6 * tgt.h
        same_col = abs(b.x0 - tgt.x0) < 25 or abs(b.cx - tgt.cx) < 25
        if same_row or same_col:
            out.append(b)
    return out


# ------------------------------------------------------------------ table reading
_NUM_RE = re.compile(r"^[-+(]?\$?[\d.,:/%\-–+()]*\d[\d.,:/%\-–+()kKmM]*$")
_DASH_RE = re.compile(r"^[-–—]{1,3}$")


def is_num(t: str) -> bool:
    t = t.strip().replace(" ", "")
    return bool(_NUM_RE.match(t) or _DASH_RE.match(t))


def clean_num(t: str) -> str:
    t = t.strip().replace(" ", "").replace("–", "-").replace("—", "-")
    return t


def _split(b: TextBox) -> List[TextBox]:
    """RapidOCR merges neighbouring numeric cells ('12 5,432'); split them back apart."""
    parts = b.text.split()
    if len(parts) <= 1 or not all(is_num(p) for p in parts):
        return [b]
    total = sum(len(p) for p in parts) + (len(parts) - 1)
    out, cursor = [], b.x0
    for p in parts:
        w = b.w * len(p) / total
        out.append(TextBox(cursor, b.y0, cursor + w, b.y1, p, b.score))
        cursor += w + b.w / total
    return out


def lines_of(boxes: Sequence[TextBox]) -> List[List[TextBox]]:
    if not boxes:
        return []
    mh = statistics.median(b.h for b in boxes)
    out: List[List[TextBox]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if out and abs(statistics.mean(x.cy for x in out[-1]) - b.cy) < 0.55 * mh:
            out[-1].append(b)
        else:
            out.append([b])
    return [sorted(l, key=lambda b: b.x0) for l in out]


def _cluster_1d(vals: List[float], tol: float) -> List[List[float]]:
    out: List[List[float]] = []
    for v in sorted(vals):
        if out and v - statistics.mean(out[-1]) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


@dataclass
class Column:
    key: str          # header text ('' if none found)
    x: float          # centre x (numeric) / left x (text)
    kind: str         # 'num' | 'text'
    group: str = ""   # super-header above, e.g. PASSING


@dataclass
class Table:
    columns: List[Column] = field(default_factory=list)
    rows: List[Dict] = field(default_factory=list)     # {"y": float, "cells": {col_index: str}}
    labels: List[Tuple[str, float, float]] = field(default_factory=list)  # (text, cx, cy) above the header
    header_y: Optional[float] = None
    first_row_y: Optional[float] = None
    pitch: Optional[float] = None
    x0: float = 0
    x1: float = REF_W

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Table":
        t = Table(**{k: v for k, v in d.items() if k not in ("columns",)})
        t.columns = [Column(**c) for c in d.get("columns", [])]
        t.rows = [{"y": r["y"], "cells": {int(k): v for k, v in r["cells"].items()}} for r in d.get("rows", [])]
        t.labels = [tuple(x) for x in d.get("labels", [])]
        return t

    @property
    def headers(self) -> List[str]:
        return [c.key for c in self.columns]


def parse_table(boxes: Sequence[TextBox], min_num_per_row: int = 2) -> Table:
    toks: List[TextBox] = []
    for b in boxes:
        toks.extend(_split(b))
    lines = lines_of(toks)
    if not lines:
        return Table()

    def nnum(line):
        return sum(1 for b in line if is_num(b.text))

    data = [l for l in lines if nnum(l) >= min_num_per_row]
    if not data:
        return Table(labels=[(b.text, b.cx, b.cy) for b in toks])

    # longest run of evenly spaced data lines = the table body
    ys = [statistics.mean(b.cy for b in l) for l in data]
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    pitch_guess = statistics.median(gaps) if gaps else 2.2 * statistics.median(b.h for b in data[0])
    runs, cur = [], [0]
    for i, g in enumerate(gaps, start=1):
        if g <= 2.6 * pitch_guess:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    run = max(runs, key=len)
    body = [data[i] for i in run]
    body_y = [ys[i] for i in run]
    rgaps = [b - a for a, b in zip(body_y, body_y[1:])]
    pitch = min(rgaps) if rgaps else pitch_guess
    if rgaps:  # median of the gaps that look like single steps
        singles = [g for g in rgaps if g < 1.5 * min(rgaps)]
        pitch = statistics.median(singles)

    # numeric columns: cluster centre-x of numbers
    nums = [b for l in body for b in l if is_num(b.text)]
    mw = statistics.median(b.w for b in nums) if nums else 30
    min_count = max(1, int(0.25 * len(body)))
    num_cols = [statistics.median(c) for c in _cluster_1d([b.cx for b in nums], max(14.0, 0.55 * mw))
                if len(c) >= min_count]
    # text columns: cluster left-x of words (names are left-aligned)
    txt_groups = []
    for l in body:  # join words of one cell ("Ohio" "State") before clustering
        ws = [b for b in l if not is_num(b.text)]
        merged: List[TextBox] = []
        for b in ws:
            between = [n for n in num_cols if merged and merged[-1].x1 - 4 < n < b.x0 + 4]
            if merged and b.x0 - merged[-1].x1 < 1.2 * b.h and not between:
                m = merged[-1]
                merged[-1] = TextBox(m.x0, min(m.y0, b.y0), b.x1, max(m.y1, b.y1), m.text + " " + b.text, m.score)
            else:
                merged.append(b)
        txt_groups.extend(merged)
    text_cols = [min(c) for c in _cluster_1d([b.x0 for b in txt_groups], 30.0) if len(c) >= min_count]
    # a text column that sits on a numeric column's centre is really that numeric column
    text_cols = [x for x in text_cols if all(abs(x - n) > 0.6 * mw for n in num_cols)]

    cols = [Column("", x, "text") for x in text_cols] + [Column("", x, "num") for x in num_cols]
    cols.sort(key=lambda c: c.x)

    # header line: nearest mostly-text line above the body
    top_y = body_y[0]
    above = [l for l in lines if statistics.mean(b.cy for b in l) < top_y - 0.45 * pitch]
    header, group_line = None, None
    for l in reversed(above):
        ly = statistics.mean(b.cy for b in l)
        if top_y - ly > 3.2 * pitch:
            break
        if len(l) >= 2 and nnum(l) <= 0.5 * len(l):
            header = l
            break
    if header:
        hy = statistics.mean(b.cy for b in header)
        hh = statistics.median(b.h for b in header)
        for l in reversed(above):
            ly = statistics.mean(b.cy for b in l)
            if ly < hy - 0.5 * hh and hy - ly < 2.6 * hh and len(l) >= 2 and nnum(l) == 0:
                group_line = l
                break
        _assign_headers(header, cols, mw)
        if group_line:
            for c in cols:
                for g in group_line:
                    cx = c.x if c.kind == "num" else c.x + 10
                    if g.x0 - 6 <= cx <= g.x1 + 6 and g.w > 1.5 * mw:
                        c.group = g.text.strip()

    # rows
    rows = []
    for l, y in zip(body, body_y):
        cells: Dict[int, str] = {}
        for b in l:
            if is_num(b.text) and num_cols:
                idx = min((i for i, c in enumerate(cols) if c.kind == "num"),
                          key=lambda i: abs(cols[i].x - b.cx), default=None)
                if idx is None or abs(cols[idx].x - b.cx) > 1.2 * mw + 10:
                    continue
                v = clean_num(b.text)
            else:
                cand = [i for i, c in enumerate(cols) if c.kind == "text" and c.x <= b.x0 + 18]
                if not cand:
                    near = [i for i, c in enumerate(cols) if c.kind == "num" and abs(c.x - b.cx) < mw]
                    if not near:
                        continue
                    idx = near[0]  # e.g. a 'T' or 'N/A' in a numeric column
                else:
                    idx = max(cand, key=lambda i: cols[i].x)
                v = b.text.strip()
            cells[idx] = (cells[idx] + " " + v) if idx in cells and cols[idx].kind == "text" else v
        rows.append({"y": float(y), "cells": cells})

    hdr_top = min(b.y0 for b in header) if header else top_y - pitch
    if group_line:
        hdr_top = min(hdr_top, min(b.y0 for b in group_line))
    labels = [(b.text, b.cx, b.cy) for b in toks if b.y1 < hdr_top - 2]
    xs = [b.x0 for l in body for b in l] + [b.x1 for l in body for b in l]
    return Table(columns=cols, rows=rows, labels=labels,
                 header_y=float(statistics.mean(b.cy for b in header)) if header else None,
                 first_row_y=float(body_y[0]), pitch=float(pitch),
                 x0=float(min(xs) - 12), x1=float(max(xs) + 12))


def fill_missing(frame: np.ndarray, t: Table, ocr) -> int:
    """Re-read cells that full-frame OCR skipped (usually single characters like 'S' or '7')."""
    if not t.rows or not t.columns or not t.pitch:
        return 0
    xs = [c.x for c in t.columns]
    num_x = sorted(c.x for c in t.columns if c.kind == "num")
    sp = statistics.median([b - a for a, b in zip(num_x, num_x[1:])]) if len(num_x) > 1 else 120
    filled = 0
    for row in t.rows:
        y0, y1 = int(row["y"] - 0.45 * t.pitch), int(row["y"] + 0.45 * t.pitch)
        for i, c in enumerate(t.columns):
            if i in row["cells"]:
                continue
            if c.kind == "num":
                cx0, cx1 = int(c.x - 0.48 * sp), int(c.x + 0.48 * sp)
            else:
                nxt = [x for x in xs if x > c.x + 5]
                right = min(nxt) - (0.5 * sp if nxt and any(cc.kind == "num" and cc.x == min(nxt) for cc in t.columns)
                                    else 8) if nxt else t.x1
                cx0, cx1 = int(c.x - 6), int(right)
            cell = frame[max(0, y0):y1, max(0, cx0):max(cx0 + 1, cx1)]
            if cell.size == 0:
                continue
            txt, score = ocr.read_short(cell)
            if not txt or score < 0.3 or (c.kind == "num" and not is_num(txt)):
                continue
            row["cells"][i] = clean_num(txt) if c.kind == "num" else txt.upper() if len(txt) <= 4 else txt
            filled += 1
    return filled


def _assign_headers(header: List[TextBox], cols: List[Column], mw: float) -> None:
    # split a header box that covers several numeric columns ("ATT CMP YDS")
    out: List[TextBox] = []
    for b in header:
        inside = [c for c in cols if c.kind == "num" and b.x0 - 4 <= c.x <= b.x1 + 4]
        parts = b.text.split()
        if len(inside) >= 2 and len(parts) == len(inside):
            for p, c in zip(parts, inside):
                out.append(TextBox(c.x - 5, b.y0, c.x + 5, b.y1, p, b.score))
        else:
            out.append(b)
    pairs = []
    for ci, c in enumerate(cols):
        for hi, b in enumerate(out):
            if c.kind == "num":
                d = 0.0 if b.x0 - 6 <= c.x <= b.x1 + 6 else min(abs(b.x0 - c.x), abs(b.x1 - c.x)) + 5
                d += abs(b.cx - c.x) * 0.01
                lim = 0.9 * mw + 12
            else:
                d = abs(b.x0 - c.x)
                lim = 60
            if d <= lim:
                pairs.append((d, ci, hi))
    done_c, done_h = set(), set()
    for d, ci, hi in sorted(pairs):  # one header per column, closest first
        if ci in done_c or hi in done_h:
            continue
        cols[ci].key = out[hi].text.strip()
        done_c.add(ci)
        done_h.add(hi)


# ------------------------------------------------------------------ cheap pixel checks
def small_gray(frame: np.ndarray, region=None, size=(240, 135)) -> np.ndarray:
    if region is not None:
        x0, y0, x1, y1 = (int(v) for v in region)
        frame = frame[y0:y1, x0:x1]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.int16)


def edge_sig(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Map of vertical text strokes, downsampled. Ignores highlight backgrounds (cursor moves)
    but changes whenever the text itself changes or moves."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    e = (np.abs(cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)) > 60).astype(np.float32)
    return cv2.resize(e, size, interpolation=cv2.INTER_AREA)


def region_edges(frame: np.ndarray, region, size=(480, 135)) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in region)
    crop = frame[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
    if crop.size == 0:
        return np.zeros((size[1], size[0]), np.float32)
    return edge_sig(crop, size)


def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


@dataclass
class Geometry:
    """Where a table sits on screen, learned once from one OCR pass."""

    x0: float
    x1: float
    first_row_y: float
    pitch: float
    slots: int                 # row slots seen on screen
    col_xs: List[float]        # numeric column centres
    header_y: Optional[float] = None
    rows_seen: int = 0

    @property
    def top(self) -> float:
        t = self.first_row_y - self.pitch / 2
        if self.header_y is not None:
            t = min(t, self.header_y - 0.6 * self.pitch)
        return max(0.0, t)

    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Header + rows: where any table movement shows up."""
        return int(self.x0), int(self.top), int(self.x1), int(self.first_row_y + (self.slots - 0.5) * self.pitch)

    @property
    def above(self) -> Tuple[int, int, int, int]:
        """Everything above the table (team name, category, view label...)."""
        return 0, 0, REF_W, max(1, int(self.top) - 2)

    @staticmethod
    def from_table(t: Table) -> Optional["Geometry"]:
        if not t.rows or not t.pitch:
            return None
        last = t.rows[-1]["y"]
        num_x = [c.x for c in t.columns if c.kind == "num"]
        slots = max(1, round((last - t.first_row_y) / t.pitch) + 1)
        return Geometry(t.x0, t.x1, t.first_row_y, t.pitch, slots, num_x, t.header_y, rows_seen=len(t.rows))

    def row_bands(self, frame: np.ndarray) -> List[np.ndarray]:
        out = []
        for i in range(self.slots):
            yc = self.first_row_y + i * self.pitch
            y0, y1 = int(yc - self.pitch * 0.42), int(yc + self.pitch * 0.42)
            band = frame[max(0, y0):max(1, y1), int(self.x0):int(self.x1)]
            if band.size == 0:
                out.append(np.zeros((12, 384), np.float32))
                continue
            out.append(edge_sig(band, (384, 12)))
        return out

    def col_bands(self, frame: np.ndarray) -> List[np.ndarray]:
        y0, y1 = int(self.first_row_y - self.pitch / 2), int(self.first_row_y + (self.slots - 0.5) * self.pitch)
        xs = sorted(self.col_xs)
        if len(xs) < 2:
            return []
        w = statistics.median([b - a for a, b in zip(xs, xs[1:])])
        out = []
        for x in xs:
            band = frame[max(0, y0):y1, max(0, int(x - 0.45 * w)):int(x + 0.45 * w)]
            if band.size == 0:
                continue
            out.append(edge_sig(band, (16, 192)))
        return out


def _outlier(colors: List[np.ndarray], min_dist: float = 30.0) -> Optional[int]:
    if len(colors) < 2:
        return None
    arr = np.array(colors, dtype=np.float32)
    med = np.median(arr, axis=0)
    d = np.linalg.norm(arr - med, axis=1)
    order = np.argsort(-d)
    if d[order[0]] < min_dist or (len(d) > 2 and d[order[1]] > 0.6 * d[order[0]] and len(d) > 3):
        return None
    return int(order[0])


def cursor_index(frame: np.ndarray, g: "Geometry", axis: str) -> Optional[int]:
    """Which visible row (axis='v') or column (axis='h') the cursor is on, from its background colour."""
    if axis == "v":
        cols = []
        for i in range(g.slots):
            yc = g.first_row_y + i * g.pitch
            band = frame[int(yc - 0.4 * g.pitch):int(yc + 0.4 * g.pitch), int(g.x0):int(g.x1)]
            cols.append(np.median(band.reshape(-1, 3), axis=0) if band.size else np.zeros(3))
        return _outlier(cols)
    xs = sorted(g.col_xs)
    if len(xs) < 2:
        return None
    sp = statistics.median([b - a for a, b in zip(xs, xs[1:])])
    strips = []
    if g.header_y is not None:
        strips.append((g.header_y - 0.35 * g.pitch, g.header_y + 0.35 * g.pitch))
    strips.append((g.first_row_y - 0.4 * g.pitch, g.first_row_y + (g.slots - 0.6) * g.pitch))
    for y0, y1 in strips:
        cols = []
        for x in xs:
            p = frame[int(y0):int(y1), int(x - 0.4 * sp):int(x + 0.4 * sp)]
            cols.append(np.median(p.reshape(-1, 3), axis=0) if p.size else np.zeros(3))
        i = _outlier(cols)
        if i is not None:
            return i
    return None


def changed_fraction(a: np.ndarray, b: np.ndarray, pix: int = 40) -> float:
    return float((np.abs(a - b) > pix).mean())


def bands_changed(a: List[np.ndarray], b: List[np.ndarray], thresh: float = 0.009) -> float:
    """Fraction of row/column bands whose text changed. Cursor moves don't count; a scroll
    changes most bands."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if float(np.abs(a[i] - b[i]).mean()) > thresh) / n
