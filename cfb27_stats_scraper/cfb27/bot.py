"""Drives College Football 27: launch -> Dynasty -> Load/Start Dynasty -> your dynasty ->
Stats & Records -> Team Stats (every view) -> Season Stats (every team x every category).

How the speed comes about:
  * Tables are read a whole page at a time: a quick burst of d-pad taps jumps one screenful of
    rows/columns, and the scan snakes (down, right, down, left...) so it never returns to the top.
  * Pixel checks (no OCR) decide whether a jump scrolled, wrapped around or hit the end.
  * Screenshots go to a pool of OCR worker processes; the controller never waits for OCR.
  * Waits are "until the screen stops changing", not fixed sleeps.
  * Per stat category the column-page count is learned on the first team and reused.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import vision as V
from .ocr import OCR, TextBox
from .pool import CaptureLog, OcrPool

log = logging.getLogger(__name__)

OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _or(a, b):
    return b if a is None else a


class NavError(RuntimeError):
    pass


def _same_item(a: TextBox, b: TextBox) -> bool:
    if a is b:
        return True
    ix = min(a.x1, b.x1) - max(a.x0, b.x0)
    return (ix > -10 and abs(a.cy - b.cy) < 1.6 * max(a.h, b.h)) or (abs(a.cx - b.cx) < 15 and abs(a.cy - b.cy) < 15)


class Bot:
    def __init__(self, cfg: dict, pad, cap, ocr: OCR, caplog: CaptureLog, pool: OcrPool,
                 debug_dir: Path, launched_now: bool = False, focus=None):
        self.cfg = cfg
        self.pad = pad
        self.cap = cap
        self.ocr = ocr
        self.caplog = caplog
        self.pool = pool
        self.debug_dir = debug_dir
        self.launched_now = launched_now
        self.focus = focus  # callable that brings the game to the front, or None
        t = cfg["table"]
        self.min_wait = float(t.get("min_wait", 0.08))
        self.change_timeout = float(t.get("change_timeout", 0.8))
        self.settle_timeout = float(t.get("settle_timeout", 1.5))
        self.change_thresh = float(t.get("change_thresh", 0.0008))  # share of pixels that must change
        self.stable_thresh = float(t.get("stable_thresh", 0.0004))
        self.menu_wait = float(cfg["menus"].get("menu_wait", 0.6))
        self.debug = bool(cfg["output"].get("debug", False))
        self.scanner = Scanner(self)
        self._shot_n = 0

    # ------------------------------------------------------------------ primitives
    def grab(self) -> np.ndarray:
        return V.to_ref(self.cap.grab())

    def ensure_focus(self) -> None:
        if self.focus:
            self.focus()

    def _sig(self, frame: np.ndarray, region=None) -> np.ndarray:
        return V.small_gray(frame, region, (480, 270))

    def _differs(self, a: np.ndarray, b: np.ndarray, thresh: float) -> bool:
        return V.changed_fraction(a, b, 30) > thresh

    def after(self, before: np.ndarray, region=None, change_timeout: Optional[float] = None
              ) -> Tuple[np.ndarray, bool]:
        """Wait for the screen to react to an input, then for it to stop moving."""
        ct = self.change_timeout if change_timeout is None else change_timeout
        ref = self._sig(before, region)
        time.sleep(self.min_wait)
        t0 = time.time()
        cur = self.grab()
        changed = self._differs(self._sig(cur, region), ref, self.change_thresh)
        while not changed and time.time() - t0 < ct:
            time.sleep(0.02)
            cur = self.grab()
            changed = self._differs(self._sig(cur, region), ref, self.change_thresh)
        if not changed:
            return cur, False
        prev = self._sig(cur, region)
        t1 = time.time()
        while time.time() - t1 < self.settle_timeout:
            time.sleep(0.03)
            cur = self.grab()
            sg = self._sig(cur, region)
            if not self._differs(sg, prev, self.stable_thresh):
                break
            prev = sg
        return cur, True

    def act(self, buttons: Sequence[str], region=None, change_timeout: Optional[float] = None,
            taps: int = 1) -> Tuple[np.ndarray, bool]:
        before = self.grab()
        self.ensure_focus()
        for i, b in enumerate(buttons):
            if taps > 1:
                self.pad.tap(b, taps)
            else:
                self.pad.press(b)
            if i < len(buttons) - 1:
                time.sleep(0.25)
        return self.after(before, region, change_timeout)

    def settled(self, timeout: float = 1.5) -> np.ndarray:
        prev = self.grab()
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.06)
            cur = self.grab()
            if not self._differs(self._sig(cur), self._sig(prev), self.stable_thresh):
                return cur
            prev = cur
        return prev

    def read(self, frame: Optional[np.ndarray] = None, region=None) -> List[TextBox]:
        return self.ocr.read(self.settled() if frame is None else frame, region)

    def shot(self, frame: np.ndarray, tag: str, force: bool = False) -> None:
        if not (self.debug or force):
            return
        self._shot_n += 1
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.debug_dir / f"{self._shot_n:03d}_{V.norm(tag)[:40]}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    def route(self, name: str) -> bool:
        """Optional fixed button route from config [routes] (used instead of text navigation)."""
        seq = self.cfg.get("routes", {}).get(name) or []
        if not seq:
            return False
        log.info("Route %s: %s", name, " ".join(seq))
        for b in seq:
            if b.upper().startswith("WAIT"):
                time.sleep(float(b[4:] or 1))
            else:
                self.act([b.upper()], change_timeout=1.2)
        return True

    # ------------------------------------------------------------------ menus
    def select_item(self, target: str, scroll: str = "DOWN", max_steps: int = 30, cutoff: float = 0.8) -> bool:
        """Move the menu cursor onto the item labelled `target` and press A."""
        no_hl, last_hl, stuck, axis_flip = 0, None, 0, False
        for _ in range(max_steps):
            f = self.settled()
            boxes = self.ocr.read(f)
            tgt = V.find_text(boxes, target, cutoff)
            hl = None
            if tgt is not None:
                sib = V.siblings(boxes, tgt)
                if sib:
                    hl = V.highlighted_box(f, boxes, [tgt] + sib)
            if hl is None:
                hl = V.highlighted_box(f, boxes)
            log.debug("select %r: target=%s highlight=%s", target, tgt, hl)
            if tgt is not None and hl is not None and _same_item(tgt, hl):
                log.info("Selecting %r", target)
                self.act(["A"], change_timeout=2.0)
                return True
            if tgt is None:
                d = scroll  # not on screen yet: scroll the list
            elif hl is None:
                no_hl += 1
                if no_hl >= 3:
                    # Can't see a cursor. If the target is the only real choice, take it.
                    log.warning("Can't see which item is highlighted; pressing A on %r anyway", target)
                    self.shot(f, f"no_highlight_{target}", force=True)
                    self.act(["A"], change_timeout=2.0)
                    return True
                d = scroll
            else:
                dx, dy = tgt.cx - hl.cx, tgt.cy - hl.cy
                vertical = abs(dy) >= abs(dx) * 0.6
                if axis_flip:
                    vertical = not vertical
                d = ("DOWN" if dy > 0 else "UP") if vertical else ("RIGHT" if dx > 0 else "LEFT")
                if last_hl is not None and _same_item(hl, last_hl):
                    stuck += 1
                    if stuck >= 2:
                        axis_flip, stuck = not axis_flip, 0
                else:
                    stuck = 0
                last_hl = hl
            self.act([d], change_timeout=self.menu_wait)
        self.shot(self.grab(), f"select_failed_{target}", force=True)
        return False

    def go_to_dynasty_hub(self) -> None:
        m = self.cfg["menus"]
        name = self.cfg["dynasty"]["name"]
        timeout = float(self.cfg["game"].get("launch_timeout_sec", 420))
        deadline = time.time() + timeout
        picked, picks = False, 0
        if self.route("to_dynasty_hub"):
            return
        while time.time() < deadline:
            self.ensure_focus()
            f = self.settled(2.0)
            boxes = self.ocr.read(f)
            hub = V.count_markers(boxes, m["hub_markers"])
            if hub >= int(m.get("hub_markers_needed", 2)) or \
                    V.count_markers(boxes, m["stats_tab_markers"]) >= len(m["stats_tab_markers"]):
                log.info("In the dynasty")
                return
            if picks < 3 and V.find_text(boxes, name, 0.92):
                log.info("Dynasty list: picking %r", name)
                picks += 1
                if self.select_item(name, cutoff=0.92):
                    picked = True
                    self.launched_now = True  # from here on, pop-ups get A
                    time.sleep(3)
                continue
            if V.find_text(boxes, m["load_item"], 0.8):
                self.select_item(m["load_item"])
                continue
            if V.count_markers(boxes, m["main_menu_markers"]) >= 2:
                log.info("Main menu")
                if not self.select_item(m["main_menu_item"]):
                    raise NavError(f"Couldn't select {m['main_menu_item']!r} on the main menu")
                continue
            # Intro, 'press any button', update notice, loading screen...
            self.shot(f, "unknown_screen")
            btn = "A" if (self.launched_now or picked) else "B"
            log.info("Unrecognised screen; pressing %s", btn)
            self.act([btn], change_timeout=2.5)
            time.sleep(1.0)
        self.shot(self.grab(), "hub_timeout", force=True)
        raise NavError("Timed out getting into the dynasty")

    def go_to_stats_tab(self) -> None:
        m = self.cfg["menus"]
        if self.route("hub_to_stats_tab"):
            return
        for _ in range(int(m.get("max_tab_presses", 10))):
            boxes = self.read()
            if V.count_markers(boxes, m["stats_tab_markers"]) >= len(m["stats_tab_markers"]):
                log.info("On Stats & Records")
                return
            _, changed = self.act([m.get("tab_next", "RB")], change_timeout=1.5)
            if not changed:  # something (a pop-up) is swallowing RB
                self.act(["A"], change_timeout=1.5)
        self.shot(self.grab(), "stats_tab_missing", force=True)
        raise NavError("Couldn't find the Stats & Records tab")

    def open_item(self, item: str, route: str) -> None:
        if self.route(route):
            return
        if not self.select_item(item):
            raise NavError(f"Couldn't open {item!r}")
        time.sleep(0.5)

    def back_to_stats_tab(self) -> None:
        m = self.cfg["menus"]
        for _ in range(5):
            boxes = self.read()
            if V.count_markers(boxes, m["stats_tab_markers"]) >= len(m["stats_tab_markers"]):
                return
            self.act(["B"], change_timeout=1.5)
        raise NavError("Couldn't get back to Stats & Records")

    # ------------------------------------------------------------------ Team Stats
    def view_label(self, views: Sequence[str]) -> Optional[str]:
        fx0, fy0, fx1, fy1 = self.cfg["team_stats"].get("label_region", [0.45, 0.0, 1.0, 0.3])
        region = (fx0 * V.REF_W, fy0 * V.REF_H, fx1 * V.REF_W, fy1 * V.REF_H)
        boxes = self.ocr.read(self.settled(), region)
        for b in boxes:
            c = V.best_choice(b.text, views, 0.8)
            if c:
                return c
        return None

    def scan_team_stats(self) -> int:
        ts = self.cfg["team_stats"]
        views: List[str] = ts["views"]
        nxt = ts.get("next_view", ["RT"])
        grids = 0
        for vi, view in enumerate(views):
            if vi > 0:
                self.act(nxt, change_timeout=1.5)
            cur = self.view_label(views)
            tries = 0
            while cur is not None and cur != view and tries < len(views):
                log.info("View is %s, want %s; pressing %s", cur, view, " ".join(nxt))
                self.act(nxt, change_timeout=1.5)
                cur = self.view_label(views)
                tries += 1
            if cur is None:
                log.warning("Couldn't read the view selector; assuming it's on %s", view)
            if ts.get("enter_table"):
                self.act(ts["enter_table"])
            t0 = time.time()
            n = self.scanner.scan(f"team:{vi}", "team", {"section": "team", "view": view, "view_idx": vi})
            log.info("Team Stats / %s: %d screenshots in %.1fs", view, n, time.time() - t0)
            grids += 1
        return grids

    # ------------------------------------------------------------------ Season Stats
    def scan_season_stats(self, n_teams: int) -> int:
        ss = self.cfg["season_stats"]
        next_team = ss.get("next_team", "LT")
        next_cat = ss.get("next_category", "RT")
        max_cat = int(ss.get("max_categories", 15))
        n_cat: Optional[int] = None
        sc = self.scanner
        very_first = None
        teams_done = 0
        for ti in range(n_teams):
            t0 = time.time()
            if ti > 0:
                f, changed = self.act([next_team], change_timeout=2.0)
                if not changed:
                    f, changed = self.act([next_team], change_timeout=2.0)
                if not changed:
                    log.warning("%s didn't change the team; stopping", next_team)
                    break
                if very_first is not None and sc.same_screen(very_first, f, "season:0"):
                    log.info("Back at the first team: all %d teams done", ti)
                    break
            team_ref = self.settled()
            if very_first is None:
                very_first = team_ref
            ci = 0
            shots = 0
            while ci < max_cat:
                meta = {"section": "season", "team_idx": ti, "cat_idx": ci}
                if ti == 0 and ci > 0 and sc.is_repeat_of(f"season:{ci}", "season:0", self.grab()):
                    n_cat = ci
                    log.info("Season Stats has %d categories", n_cat)
                    break
                shots += sc.scan(f"season:{ci}", "season", meta)
                ci += 1
                if n_cat is not None and ci >= n_cat:
                    self.act([next_cat], change_timeout=1.5)  # wrap back to the first category
                    break
                f, changed = self.act([next_cat], change_timeout=1.5)
                if not changed:
                    log.info("%s didn't change the category; treating %d as the last", next_cat, ci)
                    n_cat = n_cat or ci
                    break
                if ci > 0 and sc.same_screen(team_ref, f, "season:0"):
                    n_cat = n_cat or ci
                    break
            teams_done = ti + 1
            log.info("Season Stats team %d: %d screenshots in %.1fs (OCR queue: %d)",
                     ti + 1, shots, time.time() - t0, self.pool.pending())
        return teams_done


class Scanner:
    """Captures every page of the table on screen, as fast as the game will scroll."""

    def __init__(self, bot: Bot):
        self.bot = bot
        t = bot.cfg["table"]
        self.v_next, self.v_prev = t.get("v_next", "DOWN"), t.get("v_prev", "UP")
        self.h_next, self.h_prev = t.get("h_next", "RIGHT"), t.get("h_prev", "LEFT")
        self.overlap = int(t.get("page_overlap", 1))
        self.resets = bool(t.get("scroll_resets_on_switch", True))
        self.max_pages = int(t.get("max_pages", 40))
        self.geoms: Dict[str, Optional[V.Geometry]] = {}
        self.calib: Dict[str, V.Table] = {}
        self.hpages: Dict[str, int] = {}
        self.rows_per_page: Dict[str, int] = {}
        for fam, n in (bot.cfg["table"].get("rows_per_page") or {}).items():
            self.rows_per_page[fam] = int(n)
        self.grid = 0

    # ---------------------------------------------------------------- calibration
    def calibrate(self, kind: str, frame: np.ndarray) -> Optional[V.Geometry]:
        if self.geoms.get(kind) is None:
            t = V.parse_table(self.bot.ocr.read(frame))
            self.calib[kind] = t
            g = V.Geometry.from_table(t)
            if g is not None:  # an empty table is re-checked next time (another team may have rows)
                self.geoms[kind] = g
                log.info("Table %s: %d columns %s, %d rows visible", kind, len(t.columns),
                         [c.key for c in t.columns][:14], len(t.rows))
            return g
        return self.geoms[kind]

    def is_repeat_of(self, kind: str, first_kind: str, frame: np.ndarray) -> bool:
        """True if the table now on screen is the same stat category as `first_kind`."""
        a = self.calib.get(first_kind)
        self.calibrate(kind, frame)
        b = self.calib.get(kind)
        if a is None or b is None:
            return False
        same_hdr = bool(a.headers) and [V.norm(h) for h in a.headers] == [V.norm(h) for h in b.headers]
        la = {V.norm(x[0]) for x in a.labels}
        lb = {V.norm(x[0]) for x in b.labels}
        same_lbl = len(la & lb) >= 0.8 * max(1, len(la | lb))
        if same_hdr and (same_lbl or not la):
            self.geoms.pop(kind, None)
            self.calib.pop(kind, None)
            return True
        return False

    def same_screen(self, a: np.ndarray, b: np.ndarray, kind: str) -> bool:
        """Same team + same category on screen (used to spot the LT/RT lists wrapping around)."""
        g = self.geoms.get(kind)
        if g is not None:
            rows_same = V.bands_changed(g.row_bands(a), g.row_bands(b)) < 0.2
            top_same = V.mean_diff(V.region_edges(a, g.above), V.region_edges(b, g.above)) < 0.0015
            return rows_same and top_same
        full = (0, 0, V.REF_W, V.REF_H)
        return V.mean_diff(V.region_edges(a, full), V.region_edges(b, full)) < 0.0015

    def visible_rows(self, frame: np.ndarray, g: V.Geometry) -> int:
        """Rows with text from the top of the table down (pixel check, no OCR)."""
        n = 0
        y = g.first_row_y
        while y + g.pitch * 0.42 < V.REF_H:
            band = frame[int(y - g.pitch * 0.42):int(y + g.pitch * 0.42), int(g.x0):int(g.x1)]
            if band.size == 0 or float(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).std()) < 9.0:
                break
            n += 1
            y += g.pitch
        return n

    # ---------------------------------------------------------------- paging
    def page(self, cur: np.ndarray, g: V.Geometry, axis: str, forward: bool, taps: int,
             ref: Optional[np.ndarray]) -> Tuple[np.ndarray, str]:
        """Jump one page. Returns (frame, 'scrolled' | 'wrap' | 'end').

        'wrap' means the list wrapped around to its other end during the jump; the caller then
        makes one jump back the other way, which always lands on the last page."""
        bands = g.col_bands if axis == "h" else g.row_bands
        before = bands(cur)
        before_pos = V.cursor_index(cur, g, axis)
        frame = cur
        for _attempt in range(2):
            frame = self.jump(g, axis, forward, taps)
            if frame is None:
                return cur, "end"
            after = bands(frame)
            pos = V.cursor_index(frame, g, axis)
            backwards = before_pos is not None and pos is not None and (pos < before_pos if forward else pos > before_pos)
            frac = V.bands_changed(before, after)
            if frac >= 0.5:
                # after a scroll the layout can shift, so only trust the content check here
                if ref is not None and V.bands_changed(bands(ref), after) < 0.3:
                    return frame, "wrap"
                return frame, "scrolled"
            if backwards:
                return frame, "wrap"  # scrolled to the end, wrapped, and landed back on this page
            before, before_pos, cur = after, pos, frame  # only the cursor moved: jump again
        return frame, "end"

    def jump(self, g: V.Geometry, axis: str, forward: bool, taps: int) -> Optional[np.ndarray]:
        btn = (self.h_next if forward else self.h_prev) if axis == "h" else (self.v_next if forward else self.v_prev)
        frame, changed = self.bot.act([btn], region=g.region, taps=taps, change_timeout=0.6)
        return frame if changed else None

    def scan(self, kind: str, family: str, meta: dict) -> int:
        """Capture the whole table for one screen. Returns the number of screenshots taken."""
        self.grid += 1
        f0 = self.bot.settled()
        g = self.calibrate(kind, f0)
        shots = 0

        def save(frame, v, h):
            nonlocal shots
            m = dict(meta, kind=kind, family=family, grid=self.grid, v=v, h=h)
            path = self.bot.caplog.save(frame, m)
            self.bot.pool.submit(path, m)
            shots += 1

        save(f0, 0, 0)
        if g is None:
            return shots  # empty table / nothing to scroll

        rows_vis = self.visible_rows(f0, g)
        full_page = self.rows_per_page.get(family)
        needs_v = full_page is None or rows_vis >= full_page
        taps_v = max(1, (full_page or rows_vis or g.slots) - self.overlap)
        taps_h = max(1, len(g.col_xs) - self.overlap)
        H = self.hpages.get(kind)
        level: Dict[Tuple[int, int], np.ndarray] = {(0, 0): f0}
        v, h, cur, last = 0, 0, f0, False

        while True:
            # ---- sweep across the columns (right on even rows, left on odd rows)
            going_right = v % 2 == 0
            if going_right:
                while H is None or h < H:
                    if h >= self.max_pages:
                        break
                    cur, res = self.page(cur, g, "h", True, taps_h, level.get((v, 0)))
                    if res == "end":
                        H = h
                        break
                    if res == "wrap":
                        cur = _or(self.jump(g, "h", False, taps_h), cur)
                        h += 1
                        level[(v, h)] = cur
                        save(cur, v, h)
                        H = h
                        break
                    h += 1
                    level[(v, h)] = cur
                    save(cur, v, h)
                if kind not in self.hpages:
                    self.hpages[kind] = H or 0
                    if H:
                        log.info("Table %s has %d extra column page(s)", kind, H)
            else:
                while h > 0:
                    cur, res = self.page(cur, g, "h", False, taps_h, None)
                    if res == "end":
                        break
                    if res == "wrap":
                        cur = _or(self.jump(g, "h", True, taps_h), cur)
                        h = 1  # the jump back lands on the left-most page
                    h -= 1
                    level[(v, h)] = cur
                    save(cur, v, h)
            if last or not needs_v or v >= self.max_pages:
                break
            # ---- next page of rows
            cur, res = self.page(cur, g, "v", True, taps_v, level.get((0, h)))
            if res == "end":
                break
            if res == "wrap":
                cur = _or(self.jump(g, "v", False, taps_v), cur)
                last = True
            if family not in self.rows_per_page and v == 0:
                self.rows_per_page[family] = max(rows_vis, 1)
            v += 1
            level[(v, h)] = cur
            save(cur, v, h)

        if not self.resets and (v or h):
            self.home(g, v, h, taps_v, taps_h)
        return shots

    def home(self, g: V.Geometry, v: int, h: int, taps_v: int, taps_h: int) -> None:
        for _ in range(v + 1):
            self.bot.act([self.v_prev], region=g.region, taps=taps_v, change_timeout=0.4)
        for _ in range(h + 1):
            self.bot.act([self.h_prev], region=g.region, taps=taps_h, change_timeout=0.4)
