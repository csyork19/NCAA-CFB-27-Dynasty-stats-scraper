@echo off
REM Double-click to run, or schedule with Task Scheduler.
cd /d "%~dp0"
python run_scraper.py %*
