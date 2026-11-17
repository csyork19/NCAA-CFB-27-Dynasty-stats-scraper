"""The run itself: menus -> Team Stats -> Season Stats. Shared by run_scraper.py and the tests."""
from __future__ import annotations

import logging
import time

from . import merge

log = logging.getLogger(__name__)


def run_flow(bot, cfg: dict, start: str = "launch", only: str = "all", max_teams: int = 0) -> dict:
    m = cfg["menus"]
    pool = bot.pool
    summary: dict = {}
    t0 = time.time()
    if start == "launch":
        bot.go_to_dynasty_hub()
        bot.go_to_stats_tab()
        start = "stats"
        summary["menus_sec"] = round(time.time() - t0, 1)
    if only in ("all", "team") and start in ("stats", "team_stats"):
        t1 = time.time()
        if start == "stats":
            bot.open_item(m.get("team_stats_item", "Team Stats"), "stats_tab_to_team_stats")
        bot.scan_team_stats()
        summary["team_stats_sec"] = round(time.time() - t1, 1)
        if only == "all":
            bot.back_to_stats_tab()
            start = "stats"
    if only in ("all", "season"):
        t1 = time.time()
        if start == "stats":
            bot.open_item(m.get("season_stats_item", "Season Stats"), "stats_tab_to_season_stats")
        n = max_teams or int(cfg["season_stats"].get("num_teams", 0) or 0)
        if not n:
            pool.wait("team:", timeout=180)
            sheets = merge.build(pool.snapshot(), cfg)
            found = max((len(v) for k, v in sheets.items() if k.startswith("Team ")), default=0)
            n = found + 3 if found else 200
            log.info("Season Stats: up to %d teams (Team Stats found %d)", n, found)
        summary["season_teams"] = bot.scan_season_stats(n)
        summary["season_stats_sec"] = round(time.time() - t1, 1)
    return summary
