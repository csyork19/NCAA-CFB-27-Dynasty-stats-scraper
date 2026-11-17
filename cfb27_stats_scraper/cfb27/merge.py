"""Stitches the OCR'd screenshots back into whole tables and writes CSV / Excel.

Within one table scan, screenshots at the same row-page (v) share row positions, so cells line
up by y even if the name column scrolls away. Column pages (h) are stitched by finding the
overlap between their header sequences, checked against the cell values. Row pages are joined
by the row's name (team or player). Every cell is a vote across all the screenshots it appeared on.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import vision as V

log = logging.getLogger(__name__)


@dataclass
class Grid:
    columns: List[str] = field(default_factory=list)       # display names, in order
    rows: List[Dict[str, str]] = field(default_factory=list)
    labels: Counter = field(default_factory=Counter)        # text above the table (team, category...)
    name_col: Optional[str] = None


def _key(c: V.Column) -> str:
    return V.norm(c.key) or "?"


def _frozen_prefix(caps: List[V.Table]) -> int:
    """How many leftmost columns stay put across all column pages (e.g. a sticky TEAM column)."""
    if len(caps) < 2:
        n = 0
        for c in caps[0].columns:
            if c.kind != "text":
                break
            n += 1
        return n
    base = caps[0].columns
    n = 0
    for i, c in enumerate(base):
        ok = all(i < len(o.columns) and _key(o.columns[i]) == _key(c) and abs(o.columns[i].x - c.x) < 14
                 for o in caps[1:])
        if not ok:
            break
        n += 1
    return n


def _row_match(a: V.Table, b: V.Table) -> Dict[int, int]:
    """Row index in a -> row index in b, matched by y position."""
    out = {}
    tol = 0.4 * (a.pitch or 30)
    for i, ra in enumerate(a.rows):
        best = min(range(len(b.rows)), key=lambda j: abs(b.rows[j]["y"] - ra["y"]), default=None)
        if best is not None and abs(b.rows[best]["y"] - ra["y"]) <= tol:
            out[i] = best
    return out


def _shift(prev: V.Table, cur: V.Table, nf: int) -> int:
    """Number of scrolling columns between the left edge of `prev` and the left edge of `cur`."""
    ps = prev.columns[nf:]
    cs = cur.columns[nf:]
    if not ps:
        return 0
    rm = _row_match(prev, cur)
    best_s, best_score = len(ps), -1.0
    for s in range(1, len(ps) + 1):
        k = min(len(ps) - s, len(cs))
        if k <= 0:
            continue
        keys_ok = all(_key(ps[s + i]) == _key(cs[i]) or "?" in (_key(ps[s + i]), _key(cs[i])) for i in range(k))
        if not keys_ok:
            continue
        agree = total = 0
        for i in range(k):
            pi, ci = nf + s + i, nf + i
            for ra, rb in rm.items():
                va = prev.rows[ra]["cells"].get(pi)
                vb = cur.rows[rb]["cells"].get(ci)
                if va is not None and vb is not None:
                    total += 1
                    agree += V.norm(va) == V.norm(vb)
        if total and agree / total < 0.7:
            continue
        score = k + (agree / total if total else 0)
        if score > best_score:
            best_s, best_score = s, score
    return best_s


def assemble(entries: List[Tuple[dict, V.Table]]) -> Grid:
    by_v: Dict[int, List[Tuple[int, V.Table]]] = defaultdict(list)
    labels: Counter = Counter()
    for meta, t in entries:
        by_v[meta["v"]].append((meta["h"], t))
        for txt, _cx, _cy in t.labels:
            labels[txt.strip()] += 1
    for v in by_v:
        by_v[v].sort(key=lambda x: x[0])

    # Column layout from the row-page with the most column pages
    v_ref = max(by_v, key=lambda v: (len(by_v[v]), -v))
    caps = [t for _, t in by_v[v_ref] if t.columns]
    if not caps:
        return Grid(labels=labels)
    nf = _frozen_prefix(caps)
    # Column offsets per row-page: the left/right sweeps can stop at different scroll positions
    offsets: Dict[Tuple[int, int], int] = {}
    for v, pages in by_v.items():
        pages = [(h, t) for h, t in pages if t.columns]
        off = 0
        for i, (h, t) in enumerate(pages):
            if i > 0:
                off += _shift(pages[i - 1][1], t, nf)
            offsets[(v, h)] = off

    names: Dict[str, Counter] = defaultdict(Counter)
    order: Dict[str, float] = {}

    def col_id(v: int, h: int, idx: int) -> str:
        if idx < nf:
            return f"F{idx}"
        return f"P{offsets.get((v, h), 0) + idx - nf:04d}"

    # rows: (v, row-y) -> name + cells
    merged: Dict[str, Dict[str, Counter]] = {}
    row_order: List[str] = []
    name_idx = 0 if nf else None
    for v in sorted(by_v):
        pages = [(h, t) for h, t in by_v[v] if t.columns]
        if not pages:
            continue
        anchor_h, anchor = min(pages, key=lambda x: x[0])
        text_cols = [i for i, c in enumerate(anchor.columns) if c.kind == "text"]
        for ri, row in enumerate(anchor.rows):
            # row name: sticky column if any, else the first text column of the left-most page
            nm = None
            if name_idx is not None:
                nm = row["cells"].get(name_idx)
            elif text_cols and anchor_h == 0:
                nm = row["cells"].get(text_cols[0])
            rk = V.norm(nm) if nm else f"_v{v}_r{ri}"
            if rk not in merged:
                merged[rk] = defaultdict(Counter)
                row_order.append(rk)
            if nm:
                merged[rk]["__name__"][nm.strip()] += 1
        for h, t in pages:
            rm = _row_match(anchor, t)
            for ia, ib in rm.items():
                arow = anchor.rows[ia]
                nm = arow["cells"].get(name_idx) if name_idx is not None else \
                    (arow["cells"].get(text_cols[0]) if text_cols and anchor_h == 0 else None)
                rk = V.norm(nm) if nm else f"_v{v}_r{ia}"
                for ci, val in t.rows[ib]["cells"].items():
                    if ci >= len(t.columns):
                        continue
                    cid = col_id(v, h, ci)
                    merged[rk][cid][val.strip()] += 1
                    c = t.columns[ci]
                    disp = (c.group + " " + c.key).strip() if c.group else c.key
                    names[cid][disp or ""] += 1
                    order.setdefault(cid, (0 if cid.startswith("F") else 1, int(cid[1:])))

    cids = sorted(order, key=lambda c: order[c])
    disp_names, used = [], Counter()
    for cid in cids:
        nm = _pick_name(names[cid]) or f"Col{len(disp_names) + 1}"
        used[nm] += 1
        disp_names.append(nm if used[nm] == 1 else f"{nm} ({used[nm]})")
    rows = []
    for rk in row_order:
        cells = merged[rk]
        filled = sum(1 for c in cids if cells.get(c))
        if filled < max(2, 0.4 * len(cids)):
            continue  # cut-off row at a screen edge, or an OCR fragment
        rows.append({d: cells[c].most_common(1)[0][0] if cells.get(c) else None for c, d in zip(cids, disp_names)})
    return Grid(columns=disp_names, rows=rows, labels=labels, name_col=disp_names[0] if nf else None)


def _pick_name(c: Counter) -> str:
    """Most-voted header; among spellings that only differ in spacing, keep the spaced one."""
    groups: Dict[str, Counter] = defaultdict(Counter)
    for n, k in c.items():
        if n:
            groups[V.norm(n)][n] += k
    if not groups:
        return ""
    best = max(groups.values(), key=lambda g: sum(g.values()))
    return max(best, key=lambda n: (n.count(" "), best[n]))


# ------------------------------------------------------------------ labels -> team / category names
def _distinctive(groups: Dict[int, List[Counter]], min_share: float = 0.6) -> Dict[int, Optional[str]]:
    """For each group (a team, or a category) find the label text that appears on most of its
    screens but not on most screens of the other groups."""
    share: Dict[int, Counter] = {}
    for gid, cnts in groups.items():
        c = Counter()
        for cn in cnts:
            for t in set(V.norm(x) for x in cn):
                c[t] += 1
        share[gid] = Counter({t: n / max(1, len(cnts)) for t, n in c.items()})
    raw: Dict[int, Dict[str, str]] = {gid: {V.norm(x): x for cn in cnts for x in cn} for gid, cnts in groups.items()}
    out = {}
    for gid, sh in share.items():
        cands = []
        for t, s in sh.items():
            if s < min_share or len(t) < 3 or t.isdigit():
                continue
            others = [share[o].get(t, 0) for o in share if o != gid]
            elsewhere = sum(1 for x in others if x >= 0.5) / max(1, len(others))
            if elsewhere < 0.5 or len(share) == 1:
                cands.append((s - elsewhere, -len(t), t))
        out[gid] = raw[gid][max(cands)[2]] if cands else None
    return out


def to_number(v):
    if v is None:
        return None
    t = str(v).strip().replace(",", "")
    t2 = t[:-1] if t.endswith("%") else t
    t2 = t2.lstrip("$")
    if re.fullmatch(r"[+-]?\d+", t2):
        return int(t2)
    if re.fullmatch(r"[+-]?\d*\.\d+", t2):
        return float(t2)
    return t


def build(results: Sequence[dict], cfg: dict) -> Dict[str, List[dict]]:
    """Returns {sheet_name: [row dicts]}."""
    grids: Dict[int, List[Tuple[dict, V.Table]]] = defaultdict(list)
    for r in results:
        if r.get("table") is None:
            continue
        grids[r["meta"]["grid"]].append((r["meta"], V.Table.from_dict(r["table"])))

    sheets: Dict[str, List[dict]] = {}
    team_names: List[str] = []

    # ---------------- Team Stats: one sheet per view
    team_grids = {g: e for g, e in grids.items() if e[0][0]["section"] == "team"}
    by_view: Dict[int, List[int]] = defaultdict(list)
    for g, e in team_grids.items():
        by_view[e[0][0]["view_idx"]].append(g)
    for vi in sorted(by_view):
        view = team_grids[by_view[vi][0]][0][0]["view"]
        rows = []
        for g in sorted(by_view[vi]):
            grid = assemble(team_grids[g])
            for r in grid.rows:
                rows.append({k: to_number(v) for k, v in r.items()})
            if grid.name_col:
                team_names.extend(str(r[grid.name_col]) for r in grid.rows if r.get(grid.name_col))
        sheets[f"Team {view}"] = rows
        log.info("Team Stats / %s: %d teams x %d columns", view, len(rows), len(rows[0]) if rows else 0)
    team_names = list(dict.fromkeys(team_names))

    # ---------------- Season Stats: one sheet per category, all teams stacked
    season = {g: e for g, e in grids.items() if e[0][0]["section"] == "season"}
    assembled = {g: assemble(e) for g, e in season.items()}
    by_team: Dict[int, List[Counter]] = defaultdict(list)
    by_cat: Dict[int, List[Counter]] = defaultdict(list)
    for g, e in season.items():
        m = e[0][0]
        by_team[m["team_idx"]].append(assembled[g].labels)
        by_cat[m["cat_idx"]].append(assembled[g].labels)
    team_label = _distinctive(by_team)
    cat_label = _distinctive(by_cat)
    known_cats = cfg.get("season_stats", {}).get("category_names", [])

    def team_name(ti: int) -> str:
        lab = team_label.get(ti)
        if lab and team_names:
            best = V.best_choice(lab, team_names, 0.75)
            if best:
                return best
        if not lab and team_names:  # fall back to any label that is a known team
            for cn in by_team[ti]:
                for t in cn:
                    best = V.best_choice(t, team_names, 0.85)
                    if best:
                        return best
        return lab or f"Team {ti + 1}"

    def cat_name(ci: int) -> str:
        for cn in by_cat[ci]:
            for t in cn:
                k = V.best_choice(t, known_cats, 0.85)
                if k:
                    return k
        return cat_label.get(ci) or f"Category {ci + 1}"

    cats: Dict[int, List[dict]] = defaultdict(list)
    names_ci: Dict[int, str] = {}
    for g in sorted(season):
        m = season[g][0][0]
        ci, ti = m["cat_idx"], m["team_idx"]
        names_ci.setdefault(ci, cat_name(ci))
        tn = team_name(ti)
        for r in assembled[g].rows:
            cats[ci].append({"Team": tn, **{k: to_number(v) for k, v in r.items()}})
    used = Counter()
    for ci in sorted(cats):
        nm = names_ci[ci]
        used[nm] += 1
        key = f"Season {nm}" if used[nm] == 1 else f"Season {nm} {used[nm]}"
        sheets[key] = _dedupe(cats[ci])
        log.info("Season Stats / %s: %d player rows", nm, len(sheets[key]))
    return sheets


def _dedupe(rows: List[dict]) -> List[dict]:
    seen, out = set(), []
    for r in rows:
        k = tuple((k, str(v)) for k, v in r.items())
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "x"


def write(sheets: Dict[str, List[dict]], out_dir: Path, base: str, excel: bool = True) -> List[Path]:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    frames = {}
    for name, rows in sheets.items():
        if not rows:
            continue
        cols = list(dict.fromkeys(k for r in rows for k in r))
        df = pd.DataFrame(rows, columns=cols)
        frames[name] = df
        p = out_dir / f"{slug(name)}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    if excel and frames:
        x = out_dir / f"{slug(base)}.xlsx"
        try:
            with pd.ExcelWriter(x, engine="openpyxl") as xw:
                used = set()
                for name, df in frames.items():
                    sn = re.sub(r"[\[\]:*?/\\]", "", name)[:31]
                    while sn in used:
                        sn = sn[:29] + str(len(used) % 10)
                    used.add(sn)
                    df.to_excel(xw, sheet_name=sn, index=False)
                    ws = xw.sheets[sn]
                    ws.freeze_panes = "B2"
                    for col in ws.columns:
                        w = max(len(str(c.value or "")) for c in col[:200])
                        ws.column_dimensions[col[0].column_letter].width = min(28, max(6, w + 2))
            paths.append(x)
        except Exception as e:
            log.warning("Excel export skipped: %s", e)
    return paths
