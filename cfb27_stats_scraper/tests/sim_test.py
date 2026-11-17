"""End-to-end test against the fake game: real menus logic, real paging, real OCR, real merge.

    python tests/sim_test.py                 # sticky name column, no wrap-around
    python tests/sim_test.py --wrap --no-sticky
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from cfb27 import merge, vision as V  # noqa: E402
from cfb27.bot import Bot  # noqa: E402
from cfb27.flow import run_flow  # noqa: E402
from cfb27.ocr import OCR  # noqa: E402
from cfb27.pool import CaptureLog, OcrPool  # noqa: E402
from run_scraper import load_config  # noqa: E402
import sim_game as S  # noqa: E402


def compare(got: dict, want: dict) -> bool:
    ok = True
    for sheet, rows in want.items():
        g = got.get(sheet)
        if g is None:
            print(f"  MISSING SHEET {sheet!r} (have {list(got)})")
            ok = False
            continue
        key_cols = ["TEAM"] if sheet.startswith("Team") else ["Team", "NAME"]
        g = [{V.norm(k): v for k, v in r.items()} for r in g]  # header spacing doesn't matter
        key_cols = [V.norm(k) for k in key_cols]
        index = {tuple(V.norm(r.get(k)) for k in key_cols): r for r in g}
        wrong, missing_rows, cells = [], 0, 0
        for r in rows:
            r = {V.norm(c): v for c, v in r.items()}
            k = tuple(V.norm(r[c]) for c in key_cols)
            gr = index.get(k)
            if gr is None:
                missing_rows += 1
                continue
            for col, val in r.items():
                cells += 1
                gv = gr.get(col)
                if V.norm(str(merge.to_number(val))) != V.norm(str(gv)):
                    wrong.append((k, col, gv, val))
        extra = len(g) - (len(rows) - missing_rows)
        status = "OK " if not wrong and not missing_rows and extra == 0 else "BAD"
        print(f"  {status} {sheet:20s} rows {len(g)}/{len(rows)}  cells checked {cells}  wrong {len(wrong)}"
              f"  missing rows {missing_rows}  extra rows {extra}")
        for w in wrong[:5]:
            print("       ", w)
        ok &= status == "OK "
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--no-sticky", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for n in ("rapidocr_onnxruntime", "RapidOCR"):
        logging.getLogger(n).setLevel(logging.ERROR)

    cfg = load_config(HERE.parent / "config.toml")
    cfg["dynasty"]["name"] = "My Dynasty"
    cfg["team_stats"]["views"] = list(S.VIEWS)
    cfg["season_stats"]["category_names"] = list(S.CATS)
    cfg["table"].update(min_wait=0.0, change_timeout=0.15, settle_timeout=0.15)
    cfg["menus"]["menu_wait"] = 0.15

    game = S.FakeGame(wrap=args.wrap, sticky_name=not args.no_sticky)
    tmp = Path(tempfile.mkdtemp())
    pool = OcrPool(args.workers, threads=1)
    caplog = CaptureLog(tmp / "captures")
    bot = Bot(cfg, game, game, OCR(threads=1), caplog, pool, tmp / "debug", launched_now=True)
    t0 = time.time()
    summary = run_flow(bot, cfg)
    nav = time.time() - t0
    pool.wait()
    total = time.time() - t0
    sheets = merge.build(pool.snapshot(), cfg)
    merge.write(sheets, tmp / "out", "sim")
    pool.close()
    print(f"\nwrap={args.wrap} sticky={not args.no_sticky}: navigation {nav:.0f}s, total {total:.0f}s, "
          f"{caplog.n} screenshots, {game.presses} button presses, summary={summary}")
    ok = compare(sheets, S.expected())
    print("output:", tmp / "out")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
