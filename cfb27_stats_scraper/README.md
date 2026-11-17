# College Football 27: Dynasty Stats Scraper

Opens EA SPORTS College Football 27 on PC and loads your dynasty. It then captures:

* **Team Stats** for every team, in each view on the top-right selector: Offense, Defense, Conversions, Penalties, NIL. It scrolls through every row and every column.
* **Season Stats** for every team (LT) and every stat category (RT): Passing, Rushing, Receiving and the rest, with every player row and column.

The results are saved as CSV files and one Excel workbook.

## 1. Setup (once)

1. Install **Python 3.11+** (tick "Add to PATH").
2. Install the **ViGEmBus** driver (the virtual Xbox controller): https://github.com/nefarius/ViGEmBus/releases
3. In this folder, run `pip install -r requirements.txt`.
4. In the game, set **Display → Window Mode = Borderless/Windowed**. 1080p is best, but any 16:9 resolution works.
5. Edit `config.toml`:
   * `[dynasty] name`: the dynasty name exactly as it appears in Load/Start Dynasty.
   * If you play through the EA app or Epic instead of Steam, set `launch = "exe"` and `exe_path`.

## 2. Run

```
python run_scraper.py --dynasty "My Dynasty"
```

Don't touch the controller or keyboard while it runs. Output goes to `output\<dynasty>\<date_time>\`:

| File | Contents |
|---|---|
| `Team_Offense.csv`, `Team_Defense.csv`, ... | One row per team, every column |
| `Season_Passing.csv`, `Season_Rushing.csv`, ... | One row per player, with a `Team` column |
| `<dynasty>_<date>.xlsx` | All of the above as sheets |
| `captures\` | Every screenshot plus `index.jsonl` |
| `run_summary.json` | Timings, row counts and any error |

Useful options:

| Command | What it does |
|---|---|
| `--start-at stats` | The game is already on the Stats & Records tab (skips launching and menus) |
| `--start-at team_stats` / `season_stats` | The game is already on that screen |
| `--only team` / `--only season` | Do just one half |
| `--max-teams 3` | Quick Season Stats test |
| `--probe` | Reads the current screen and saves `output\probe\probe_boxes.png` and `probe.txt` showing what it found |
| `--reparse <run folder>` | Re-reads the saved screenshots and rebuilds the CSV/Excel files without the game |
| `-v` | Detailed log (`output\logs\`) |

## 3. Why it's fast

There are about 136 teams × ~8 categories, roughly 1,000+ tables, so the speed comes from removing waiting:

1. **Page jumps instead of single steps.** One burst of quick d-pad taps moves a whole screenful of rows or columns. The scan also snakes (across, down, back across) so it never returns to the top.
2. **OCR runs in parallel and never blocks the controller.** The navigation loop only saves a screenshot (about 15 ms) and moves on. A pool of worker processes reads the text in the background. Most OCR is done by the time navigation finishes.
3. **No fixed sleeps.** After each input it waits only until the screen reacts and stops moving, which is usually 0.1–0.3 s.
4. **Pixel checks instead of OCR when deciding what to do.** Comparing text strokes between frames tells it whether a jump scrolled, only moved the cursor, wrapped around, or hit the end.
5. **Learned once, reused.** Each category's column count is learned on the first team. Short tables (kicking, punting) are detected from pixels and skip vertical paging. The team count comes from Team Stats, so Season Stats knows when to stop.

Expect roughly **15–30 minutes for a full dynasty**, mostly limited by how fast the game's menus animate. `captures\` keeps everything, so if anything looks off you can run `--reparse` without replaying the game.

## 4. If a step doesn't work on your screens

All screen checks work by reading text, so most fixes are text edits in `config.toml`:

| Symptom | Fix |
|---|---|
| Stuck on main menu / sub menu | Check `main_menu_item`, `load_item` and `main_menu_markers` against the exact words on screen |
| Doesn't pick your dynasty | `[dynasty] name` must match the card text |
| Never finds Stats & Records | Set `stats_tab_markers` to two labels that only appear on that tab |
| Team Stats view doesn't change | Set `next_view` to the right buttons, e.g. `["Y"]` or `["X", "DOWN", "A"]` |
| Table doesn't scroll | Use `v_next = "LS_DOWN"` / `h_next = "LS_RIGHT"` (stick), or set `enter_table` if the cursor starts outside the table |
| Moves get skipped | Raise `tap_hold` / `tap_gap` (e.g. 0.06) |
| Animated background makes it slow | Raise `change_thresh` / `stable_thresh` |
| Menu navigation just won't work | Record a fixed button route in `[routes]` and it's used instead |

Run `--probe` on any screen that misbehaves. Send me `probe_boxes.png` and `probe.txt` and I can adjust it.

## 5. Offline test

`python tests/sim_test.py` (and `--wrap --no-sticky`) runs the whole flow against a simulated CFB-style game, using real OCR, and checks every cell.
