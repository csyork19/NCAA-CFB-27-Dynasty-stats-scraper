"""A fake College-Football-style game for offline testing: menus, a pop-up, Team Stats with
several views, and Season Stats with LT (team) / RT (category). It acts as both the gamepad
and the screen capturer, so the real bot code can drive it."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEAMS = ["Alabama", "Ohio State", "Georgia", "Texas", "Michigan", "Oregon", "Penn State", "Notre Dame",
         "LSU", "Clemson", "Florida State", "USC", "Tennessee", "Ole Miss", "Miami", "Oklahoma",
         "Kansas State", "Wisconsin"]
VIEWS = {
    "Offense": ["G", "PTS", "PPG", "YDS", "YPG", "PASS YDS", "PASS TD", "RUSH YDS", "RUSH TD", "1ST DN",
                "TOP", "PLAYS", "YPP", "3RD %", "RZ %", "TO", "SACKS", "LNG"],
    "Defense": ["G", "PA", "PAPG", "YDS A", "YPG A", "PASS A", "RUSH A", "SACK", "INT", "FF", "FR", "DEF TD",
                "TFL", "3RD %", "RZ %", "TAKE"],
    "Conversions": ["3RD ATT", "3RD CONV", "3RD %", "4TH ATT", "4TH CONV", "4TH %", "RZ ATT", "RZ TD",
                    "RZ FG", "RZ %", "2PT ATT", "2PT CONV"],
}
CATS = {
    "Passing": (["POS", "COMP", "ATT", "PCT", "YDS", "AVG", "TD", "INT", "LNG", "SCK", "RTG", "YPG", "1ST",
                 "20+"], 3),
    "Rushing": (["POS", "ATT", "YDS", "AVG", "TD", "LNG", "FUM", "YPG", "BTK"], 8),
    "Receiving": (["POS", "REC", "YDS", "AVG", "TD", "LNG", "DROP", "YAC", "YPG"], 12),
    "Defense": (["POS", "TKL", "SOLO", "TFL", "SACK", "INT", "PD", "FF", "FR", "TD"], 22),
}
FIRST = list("ABCDEGJKLMNPRSTW")
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Moore", "Taylor",
        "Thomas", "Jackson", "White", "Harris", "Martin", "Clark", "Lewis", "Walker", "Young", "Allen",
        "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Hill"]
POS = {"Passing": ["QB"], "Rushing": ["HB", "QB", "FB"], "Receiving": ["WR", "TE", "HB"],
       "Defense": ["LB", "CB", "S", "DT", "EDGE"]}

GOLD, BG, PANEL, WHITE, OLIVE, YEL = (232, 178, 38), (16, 20, 30), (30, 36, 50), (238, 238, 242), (70, 88, 40), (240, 220, 60)


def _font(size):
    for p in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


F_SM, F_MD, F_LG, F_XL = _font(22), _font(28), _font(40), _font(56)


def team_value(view: str, t: int, c: int) -> str:
    h = VIEWS[view][c]
    seed = (t * 31 + c * 17 + len(view) * 7)
    if h.endswith("%"):
        return f"{20 + seed % 70}.{seed % 10}"
    if h == "TOP":
        return f"{28 + seed % 6}:{seed % 60:02d}"
    if "YDS" in h or h.startswith("YDS"):
        return f"{1 + seed % 5},{seed * 7 % 1000:03d}"
    return str(seed % 97)


def player_name(cat: str, t: int, i: int) -> str:
    return f"{FIRST[(i + t) % len(FIRST)]}. {LAST[(i * 3 + t + len(cat)) % len(LAST)]}"


def n_players(cat: str, t: int) -> int:
    base = CATS[cat][1]
    return max(1, base + (t % 3) - 1)


def player_value(cat: str, t: int, i: int, c: int) -> str:
    h = CATS[cat][0][c]
    if h == "POS":
        p = POS[cat]
        return p[(i + t) % len(p)]
    seed = t * 13 + i * 29 + c * 11 + len(cat)
    if h in ("PCT", "AVG", "RTG", "YPG"):
        return f"{seed % 90 + 1}.{seed % 10}"
    return str(seed % 250)


class FakeGame:
    VR, VC = 12, 8  # visible table rows / numeric columns

    def __init__(self, wrap: bool = False, sticky_name: bool = True, dynasty: str = "My Dynasty"):
        self.wrap = wrap
        self.sticky = sticky_name
        self.screen = "intro"
        self.cursor = 0
        self.tab = 0
        self.tile = 0
        self.view = 0
        self.team = 0
        self.cat = 0
        self.dynasty = dynasty
        self.presses = 0
        self._reset_table()

    # ---------------- menus
    MAIN = ["Play Now", "Road to Glory", "Dynasty", "College Ultimate Team", "Team Builder", "Settings"]
    SUB = ["Load/Start Dynasty", "Join Online Dynasty", "Dynasty Settings"]
    TABS = ["Dynasty Hub", "Team", "Recruiting", "Stats & Records", "Coach"]
    TILES = ["Team Stats", "Season Stats", "Records", "Awards"]

    @property
    def dyn_list(self):
        return ["Create New Dynasty", "Gators Rebuild", self.dynasty, "Online League 2"]

    def _reset_table(self):
        self.r = self.c = 0
        self.row_off = self.col_off = 0

    def _table_dims(self) -> Tuple[int, int]:
        if self.screen == "team_stats":
            return len(TEAMS), len(VIEWS[list(VIEWS)[self.view]])
        cat = list(CATS)[self.cat]
        return n_players(cat, self.team), len(CATS[cat][0])

    def _move_table(self, d: str):
        n, m = self._table_dims()
        if d == "DOWN":
            self.r = (self.r + 1) % n if self.wrap else min(n - 1, self.r + 1)
        elif d == "UP":
            self.r = (self.r - 1) % n if self.wrap else max(0, self.r - 1)
        elif d == "RIGHT":
            self.c = (self.c + 1) % m if self.wrap else min(m - 1, self.c + 1)
        elif d == "LEFT":
            self.c = (self.c - 1) % m if self.wrap else max(0, self.c - 1)
        if self.r < self.row_off:
            self.row_off = self.r
        if self.r >= self.row_off + self.VR:
            self.row_off = self.r - self.VR + 1
        if self.c < self.col_off:
            self.col_off = self.c
        if self.c >= self.col_off + self.VC:
            self.col_off = self.c - self.VC + 1

    # ---------------- gamepad interface
    def press(self, b: str, hold=None):
        self.presses += 1
        b = b.upper()
        s = self.screen
        if s == "intro":
            if b == "A":
                self.screen, self.cursor = "main", 0
        elif s in ("main", "sub", "list"):
            items = {"main": self.MAIN, "sub": self.SUB, "list": self.dyn_list}[s]
            if b in ("DOWN", "UP"):
                self.cursor = (self.cursor + (1 if b == "DOWN" else -1)) % len(items)
            elif b == "A":
                it = items[self.cursor]
                if s == "main" and it == "Dynasty":
                    self.screen, self.cursor = "sub", 0
                elif s == "sub" and it == "Load/Start Dynasty":
                    self.screen, self.cursor = "list", 0
                elif s == "list" and it == self.dynasty:
                    self.screen = "popup"
            elif b == "B":
                self.screen = {"main": "main", "sub": "main", "list": "sub"}[s]
                self.cursor = 0
        elif s == "popup":
            if b == "A":
                self.screen, self.tab = "hub", 0
        elif s == "hub":
            if b == "RB":
                self.tab = (self.tab + 1) % len(self.TABS)
            elif b == "LB":
                self.tab = (self.tab - 1) % len(self.TABS)
            elif self.tab == 3 and b in ("RIGHT", "LEFT", "DOWN", "UP"):
                col, row = self.tile % 2, self.tile // 2
                col = min(1, col + 1) if b == "RIGHT" else max(0, col - 1) if b == "LEFT" else col
                row = min(1, row + 1) if b == "DOWN" else max(0, row - 1) if b == "UP" else row
                self.tile = row * 2 + col
            elif self.tab == 3 and b == "A":
                if self.TILES[self.tile] == "Team Stats":
                    self.screen, self.view = "team_stats", 0
                    self._reset_table()
                elif self.TILES[self.tile] == "Season Stats":
                    self.screen, self.team, self.cat = "season", 0, 0
                    self._reset_table()
        elif s in ("team_stats", "season"):
            if b in ("UP", "DOWN", "LEFT", "RIGHT"):
                self._move_table(b)
            elif b == "B":
                self.screen = "hub"
            elif s == "team_stats" and b == "RT":
                self.view = (self.view + 1) % len(VIEWS)
                self._reset_table()
            elif s == "season" and b == "RT":
                self.cat = (self.cat + 1) % len(CATS)
                self._reset_table()
            elif s == "season" and b == "LT":
                self.team = (self.team + 1) % len(TEAMS)
                self._reset_table()

    def tap(self, b, times, hold=None, gap=None):
        for _ in range(times):
            self.press(b)

    def sequence(self, names, pause=0.0):
        for n in names:
            self.press(n)

    def close(self):
        pass

    # ---------------- capturer interface
    def grab(self) -> np.ndarray:
        im = Image.new("RGB", (1920, 1080), BG)
        d = ImageDraw.Draw(im)
        s = self.screen
        if s == "intro":
            d.text((620, 480), "COLLEGE FOOTBALL 27", font=F_XL, fill=WHITE)
            d.text((760, 600), "Press A to continue", font=F_MD, fill=WHITE)
        elif s in ("main", "sub", "list"):
            title = {"main": "MAIN MENU", "sub": "DYNASTY", "list": "Load/Start Dynasty"}[s]
            items = {"main": self.MAIN, "sub": self.SUB, "list": self.dyn_list}[s]
            d.text((120, 80), title, font=F_LG, fill=WHITE)
            for i, it in enumerate(items):
                y = 220 + i * 95
                if i == self.cursor:
                    d.rectangle((100, y - 18, 760, y + 58), fill=GOLD)
                    d.text((130, y), it, font=F_MD, fill=(15, 15, 15))
                else:
                    d.rectangle((100, y - 18, 760, y + 58), fill=PANEL)
                    d.text((130, y), it, font=F_MD, fill=WHITE)
            if s == "list":
                d.text((900, 260), "Week 6  |  2026 Season", font=F_SM, fill=WHITE)
        elif s == "popup":
            d.rectangle((560, 330, 1360, 750), fill=PANEL)
            d.text((620, 380), "Dynasty Update", font=F_LG, fill=WHITE)
            d.text((620, 470), "New features were added to your dynasty.", font=F_SM, fill=WHITE)
            d.rectangle((620, 640, 860, 700), fill=GOLD)
            d.text((650, 652), "Continue", font=F_MD, fill=(15, 15, 15))
        elif s == "hub":
            x = 120
            for i, t in enumerate(self.TABS):
                w = d.textlength(t, font=F_MD)
                if i == self.tab:
                    d.rectangle((x - 16, 60, x + w + 16, 120), fill=GOLD)
                    d.text((x, 72), t, font=F_MD, fill=(15, 15, 15))
                else:
                    d.text((x, 72), t, font=F_MD, fill=WHITE)
                x += w + 80
            if self.tab == 3:
                for i, t in enumerate(self.TILES):
                    x0, y0 = 160 + (i % 2) * 700, 260 + (i // 2) * 300
                    sel = i == self.tile
                    d.rectangle((x0, y0, x0 + 620, y0 + 240), fill=GOLD if sel else PANEL)
                    d.text((x0 + 40, y0 + 160), t, font=F_LG, fill=(15, 15, 15) if sel else WHITE)
            else:
                d.text((160, 400), self.TABS[self.tab] + " overview", font=F_LG, fill=WHITE)
        elif s == "team_stats":
            view = list(VIEWS)[self.view]
            d.text((100, 60), "TEAM STATS", font=F_LG, fill=WHITE)
            d.text((1500, 70), "LT", font=F_SM, fill=WHITE)
            d.text((1560, 66), view, font=F_MD, fill=WHITE)
            d.text((1780, 70), "RT", font=F_SM, fill=WHITE)
            self._draw_table(d, ["TEAM"], [[t] for t in TEAMS], VIEWS[view],
                             lambda ri, ci: team_value(view, ri, ci))
        elif s == "season":
            cat = list(CATS)[self.cat]
            d.text((100, 40), "SEASON STATS", font=F_MD, fill=WHITE)
            d.text((100, 90), "LT", font=F_SM, fill=WHITE)
            d.text((150, 84), TEAMS[self.team], font=F_LG, fill=WHITE)
            d.text((1400, 90), cat, font=F_LG, fill=WHITE)
            d.text((1700, 96), "RT", font=F_SM, fill=WHITE)
            n = n_players(cat, self.team)
            self._draw_table(d, ["NAME"], [[player_name(cat, self.team, i)] for i in range(n)], CATS[cat][0],
                             lambda ri, ci: player_value(cat, self.team, ri, ci))
        d.text((100, 1020), "A Select    B Back", font=F_SM, fill=(150, 150, 160))
        return np.asarray(im)[:, :, ::-1].copy()

    def _draw_table(self, d, fixed_hdr, fixed_vals, headers, val):
        n = len(fixed_vals)
        top, pitch, name_x, col0, cw = 260, 56, 110, 560, 150
        # header
        d.rectangle((90, top - 70, 1830, top - 20), fill=PANEL)
        show_name = self.sticky or self.col_off == 0
        if show_name:
            d.text((name_x, top - 60), fixed_hdr[0], font=F_SM, fill=WHITE)
        start_x = col0 if show_name else name_x + 60
        for v in range(self.VC):
            ci = self.col_off + v
            if ci >= len(headers):
                break
            cx = start_x + v * cw + cw // 2
            hdr = headers[ci]
            if ci == self.c:
                d.rectangle((cx - cw // 2 + 4, top - 68, cx + cw // 2 - 4, top - 22), fill=YEL)
            w = d.textlength(hdr, font=F_SM)
            d.text((cx - w / 2, top - 60), hdr, font=F_SM, fill=(15, 15, 15) if ci == self.c else WHITE)
        for rv in range(self.VR):
            ri = self.row_off + rv
            if ri >= n:
                break
            y = top + rv * pitch
            if ri == self.r:
                d.rectangle((90, y - 6, 1830, y + 40), fill=OLIVE)
            if show_name:
                d.text((name_x, y), fixed_vals[ri][0], font=F_SM, fill=WHITE)
            for v in range(self.VC):
                ci = self.col_off + v
                if ci >= len(headers):
                    break
                cx = start_x + v * cw + cw // 2
                t = val(ri, ci)
                w = d.textlength(t, font=F_SM)
                d.text((cx - w / 2, y), t, font=F_SM, fill=WHITE)


def expected() -> Dict[str, List[dict]]:
    """Ground truth in the same shape as merge.build() output (values as text)."""
    out: Dict[str, List[dict]] = {}
    for view, hdrs in VIEWS.items():
        out[f"Team {view}"] = [{"TEAM": t, **{h: team_value(view, ti, ci) for ci, h in enumerate(hdrs)}}
                               for ti, t in enumerate(TEAMS)]
    for cat, (hdrs, _) in CATS.items():
        rows = []
        for ti, t in enumerate(TEAMS):
            for i in range(n_players(cat, ti)):
                rows.append({"Team": t, "NAME": player_name(cat, ti, i),
                             **{h: player_value(cat, ti, i, ci) for ci, h in enumerate(hdrs)}})
        out[f"Season {cat}"] = rows
    return out
