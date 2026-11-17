"""College Football 27 dynasty stats scraper.

    python run_scraper.py --dynasty "My Dynasty"         full run (launches the game if needed)
    python run_scraper.py --start-at stats               you're already on the Stats & Records tab
    python run_scraper.py --only team                    just Team Stats (or --only season)
    python run_scraper.py --max-teams 3                  quick test of Season Stats
    python run_scraper.py --probe                        read the current screen and explain what it sees
    python run_scraper.py --reparse output\\My_Dynasty\\2026-11-13_1030   rebuild CSV/Excel from saved screenshots
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore


def load_config(path: Path) -> dict:
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    for k in ("game", "dynasty", "menus", "team_stats", "season_stats", "table", "routes", "capture", "ocr",
              "output"):
        cfg.setdefault(k, {})
    return cfg


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{dt.datetime.now():%Y-%m-%d_%H%M%S}.log"
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    for noisy in ("PIL", "RapidOCR", "rapidocr", "rapidocr_onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def finish(cfg: dict, results, run_dir: Path, base: str) -> dict:
    from cfb27 import merge

    sheets = merge.build(results, cfg)
    paths = merge.write(sheets, run_dir, base, excel=cfg["output"].get("excel", True))
    return {"sheets": {k: len(v) for k, v in sheets.items()}, "files": [str(p) for p in paths]}


def reparse(cfg: dict, folder: Path) -> int:
    from cfb27.pool import OcrPool, load_index

    cap_dir = folder / "captures" if (folder / "captures").exists() else folder
    idx = load_index(cap_dir)
    pool = OcrPool(int(cfg["ocr"].get("workers", 0)), int(cfg["ocr"].get("threads_per_worker", 2)))
    t0 = time.time()
    for rec in idx:
        meta = {k: v for k, v in rec.items() if k != "file"}
        pool.submit(str(cap_dir / rec["file"]), meta)
    while pool.pending():
        print(f"\rOCR {len(idx) - pool.pending()}/{len(idx)}", end="", flush=True)
        time.sleep(1)
    print(f"\rOCR {len(idx)}/{len(idx)} in {time.time() - t0:.0f}s")
    out = finish(cfg, pool.snapshot(), folder, folder.parent.name + "_" + folder.name)
    pool.close()
    print(json.dumps(out, indent=2))
    return 0


def probe(bot, out: Path) -> None:
    """Explain what the script sees on the current screen (for setting up config.toml)."""
    import cv2

    from cfb27 import vision as V

    f = bot.settled()
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "probe.png"), f)
    boxes = bot.ocr.read(f)
    hl = V.highlighted_box(f, boxes)
    t = V.parse_table(boxes)
    lines = [f"{len(boxes)} text boxes:"] + [f"  {b.text!r:40s} at ({b.cx:.0f},{b.cy:.0f})" for b in boxes]
    lines.append(f"Highlighted item: {hl.text if hl else 'none found'}")
    if t.rows:
        lines.append(f"Table: {len(t.columns)} columns {[c.key for c in t.columns]}")
        lines.append(f"       {len(t.rows)} rows, first row y={t.first_row_y:.0f}, row pitch={t.pitch:.1f}")
        for r in t.rows[:3]:
            lines.append("       " + " | ".join(r["cells"].get(i, "") for i in range(len(t.columns))))
        lines.append(f"Text above the table: {[x[0] for x in t.labels]}")
    else:
        lines.append("No table found on this screen.")
    vis = f.copy()
    for b in boxes:
        cv2.rectangle(vis, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)),
                      (0, 0, 255) if b is hl else (0, 200, 0), 2)
    cv2.imwrite(str(out / "probe_boxes.png"), vis)
    (out / "probe.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out / 'probe.png'}, probe_boxes.png and probe.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--dynasty", help="dynasty name as shown in Load/Start Dynasty (overrides config)")
    ap.add_argument("--start-at", choices=["launch", "stats", "team_stats", "season_stats"], default="launch",
                    help="where the game already is")
    ap.add_argument("--only", choices=["all", "team", "season"], default="all")
    ap.add_argument("--max-teams", type=int, help="Season Stats: stop after N teams (testing)")
    ap.add_argument("--no-launch", action="store_true", help="fail instead of starting the game")
    ap.add_argument("--probe", action="store_true", help="describe the current screen and exit")
    ap.add_argument("--reparse", metavar="RUN_FOLDER", help="rebuild outputs from a previous run's screenshots")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.dynasty:
        cfg["dynasty"]["name"] = args.dynasty
    out_root = Path(cfg["output"].get("dir", "output"))
    if not out_root.is_absolute():
        out_root = HERE / out_root

    if args.reparse:
        setup_logging(out_root / "logs", args.verbose)
        return reparse(cfg, Path(args.reparse))

    from cfb27 import merge
    from cfb27 import window as W
    from cfb27.bot import Bot, NavError
    from cfb27.flow import run_flow
    from cfb27.gamepad import Gamepad
    from cfb27.ocr import OCR
    from cfb27.pool import CaptureLog, OcrPool

    dyn = cfg["dynasty"].get("name", "dynasty")
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = out_root / merge.slug(dyn) / stamp
    log_file = setup_logging(out_root / "logs", args.verbose)
    log = logging.getLogger("run")

    W.set_dpi_aware()
    title = cfg["game"].get("window_title", "College Football 27")
    hwnd = W.find_window(title)
    launched = False
    if hwnd is None:
        if args.no_launch or cfg["game"].get("launch", "steam") == "none":
            log.error("Game window %r not found", title)
            return 2
        W.launch(cfg["game"])
        launched = True
        t0 = time.time()
        while hwnd is None and time.time() - t0 < float(cfg["game"].get("launch_timeout_sec", 420)):
            time.sleep(5)
            hwnd = W.find_window(title)
        if hwnd is None:
            log.error("Game window never appeared (check [game].window_title)")
            return 3
        log.info("Game window found; waiting for it to start up")
        time.sleep(float(cfg["game"].get("startup_wait_sec", 20)))

    W.focus(hwnd)

    def focus():
        if not W.is_foreground(hwnd):
            W.focus(hwnd)

    ocr = OCR(threads=2)
    t = cfg["table"]
    pad = Gamepad(hold=0.06, gap=float(t.get("tap_gap", 0.035)), tap_hold=float(t.get("tap_hold", 0.035)))
    cap = W.Capturer(hwnd, cfg["capture"].get("backend", "dxcam"))
    if args.probe:
        bot = Bot(cfg, pad, cap, ocr, None, None, run_dir, launched, focus)
        probe(bot, out_root / "probe")
        pad.close()
        return 0

    pool = OcrPool(int(cfg["ocr"].get("workers", 0)), int(cfg["ocr"].get("threads_per_worker", 2)))
    caplog = CaptureLog(run_dir / "captures", int(cfg["capture"].get("jpeg_quality", 92)))
    bot = Bot(cfg, pad, cap, ocr, caplog, pool, run_dir / "debug", launched, focus)
    summary = {"dynasty": dyn, "started": dt.datetime.now().isoformat(), "log": str(log_file)}
    t_start = time.time()
    try:
        summary.update(run_flow(bot, cfg, args.start_at, args.only, args.max_teams or 0))
        summary["status"] = "ok"
        if cfg["game"].get("close_game_when_done"):
            W.close_window(hwnd)
    except (NavError, KeyboardInterrupt, Exception) as e:
        log.exception("Run stopped: %s", e)
        bot.shot(bot.grab(), "error", force=True)
        summary.update(status="error", error=repr(e))
    finally:
        pad.close()
        caplog.close()
        summary["navigation_sec"] = round(time.time() - t_start)
        log.info("Navigation done in %ds; finishing OCR on %d screenshots...", summary["navigation_sec"],
                 pool.pending())
        pool.wait()
        summary["total_sec"] = round(time.time() - t_start)
        summary.update(finish(cfg, pool.snapshot(), run_dir, f"{merge.slug(dyn)}_{stamp}"))
        pool.close()
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Done in %ds. Output: %s", summary["total_sec"], run_dir)
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
