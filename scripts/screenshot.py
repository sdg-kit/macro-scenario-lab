"""Capture dashboard screenshots for the README into docs/screenshots/.

    streamlit run app.py --server.port 8765 --server.headless true   # in another shell
    python scripts/screenshot.py [http://localhost:8765]

Requires playwright (requirements-dev.txt) and `python -m playwright install chromium`.
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8765'
OUT = Path(__file__).resolve().parents[1] / 'docs' / 'screenshots'
OUT.mkdir(parents=True, exist_ok=True)
TABS = [('Scenario dashboard', 'scenario_dashboard'), ('Housing float', 'housing_float'),
        ('Monte Carlo', 'monte_carlo'), ('Portfolio strategies', 'portfolio_strategies'),
        ('Historical data', 'historical_data')]


def settle(page, seconds=3.0):
    page.wait_for_selector('.js-plotly-plot', state='attached', timeout=60000)
    time.sleep(seconds)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1600, 'height': 1900}, device_scale_factor=1.5)
    page.goto(URL, wait_until='networkidle')
    settle(page, 6)
    for label, slug in TABS:
        page.get_by_role('tab', name=label).click()
        settle(page)
        page.screenshot(path=str(OUT / f'{slug}.png'))
        print('wrote', OUT / f'{slug}.png')
    browser.close()
