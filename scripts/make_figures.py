"""Generate the README figures into docs/figures/.

    python scripts/make_figures.py

Every figure is produced from the same engine code the dashboard runs, with the
default seed, so the images are reproducible.  Requires matplotlib (see
requirements-dev.txt).  The historical panel needs data/cache/ to be populated
(run the dashboard's "Download / refresh" button or data.historical.fetch_all()).
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theme
from theme import (ASSET_COLOR, SCENARIO_COLOR, MACRO_COLOR, PRETTY, NEUTRAL, BLUE, ORANGE,
                   AQUA, RED, TEXT_SECONDARY, DIVERGING)
from models.engine import ASSETS, SCENARIOS, simulate, summarize, portfolio, macro_path
from models.housing_float import HousingFloatConfig, simulate_housing_float

OUT = ROOT / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
theme.mpl_style()
MONTHS, PATHS, SEED = 36, 2000, 42


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path.relative_to(ROOT))


def end_label(ax, x, y, text, color, dx=4):
    ax.annotate(text, (x[-1], y[-1]), xytext=(dx, 0), textcoords='offset points',
                va='center', ha='left', fontsize=9, color=color)


# 1. Named scenarios: prescribed housing paths and the macro state they imply --------------
def fig_scenarios():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for name, sc in SCENARIOS.items():
        m = macro_path(sc, MONTHS)
        c = SCENARIO_COLOR[name]
        axes[0].plot(m.month, m.housing, color=c, label=name)
        axes[1].plot(m.month, m.unemployment, color=c)
        axes[2].plot(m.month, m.credit_bps, color=c)
    axes[0].axhline(100, color=TEXT_SECONDARY, lw=0.8, ls=':')
    axes[0].set_title('House price index (prescribed input path)')
    axes[1].set_title('Unemployment, % (derived hump)')
    axes[2].set_title('HY credit spread, bps (derived hump)')
    for ax in axes:
        ax.set_xlabel('Month')
        ax.set_xlim(0, MONTHS)
    axes[0].legend(loc='lower left', ncol=1)
    fig.suptitle('Five named scenarios: the housing trough is prescribed; labour and credit stress hump around it',
                 x=0.01, y=1.04, ha='left', fontsize=12, fontweight='semibold')
    save(fig, 'scenarios.png')


# 2. Macro path for the default scenario, small multiples ----------------------------------
def fig_macro():
    sc = SCENARIOS['Financial crisis']
    m = macro_path(sc, MONTHS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    panels = [(['housing'], 'House price index (initial = 100)', 'Index'),
              (['unemployment', 'inflation'], 'Labour and inflation', 'Percent'),
              (['fed', 'treasury', 'mortgage'], 'Policy, Treasury and mortgage rates', 'Percent'),
              (['stimulus', 'qe', 'swf'], 'Cumulative policy money (eased roll-out from month 9)', '$ trillion')]
    for ax, (cols, title, ylab) in zip(axes.flat, panels):
        for c in cols:
            ax.plot(m.month, m[c], color=MACRO_COLOR[c])
            end_label(ax, m.month.values, m[c].values, PRETTY[c], MACRO_COLOR[c])
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlim(0, MONTHS + 11)
        ax.set_xticks(range(0, MONTHS + 1, 6))
    axes[0, 0].axvline(sc.bottom_month, color=TEXT_SECONDARY, lw=0.8, ls=':')
    axes[0, 0].text(sc.bottom_month + 0.5, 99, 'trough', color=TEXT_SECONDARY, fontsize=9, va='top')
    for ax in axes[1]:
        ax.set_xlabel('Month')
    fig.suptitle('Deterministic macro path, "Financial crisis" scenario (40% housing decline, trough at month 24)',
                 x=0.01, ha='left', fontsize=12, fontweight='semibold')
    save(fig, 'macro_path.png')


# 3. Monte Carlo fan charts ----------------------------------------------------------------
def fig_fan(prices):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True)
    for ax, asset in zip(axes, ['SP500', 'BTC', 'BONDS_10Y']):
        j = ASSETS.index(asset)
        w = prices[:, :, j] / prices[:, 0:1, j] * 100
        x = np.arange(w.shape[1])
        q = np.quantile(w, [.05, .25, .5, .75, .95], axis=0)
        c = ASSET_COLOR[asset]
        ax.fill_between(x, q[0], q[4], color=c, alpha=.18, lw=0, label='5th to 95th percentile')
        ax.fill_between(x, q[1], q[3], color=c, alpha=.32, lw=0, label='25th to 75th percentile')
        ax.plot(x, q[2], color=c, lw=2.2, label='Median')
        ax.axhline(100, color=TEXT_SECONDARY, lw=0.8, ls=':')
        if asset == 'BTC':
            ax.set_yscale('log')
            ax.set_yticks([25, 50, 100, 200, 400, 800, 1600])
            ax.set_yticklabels(['25', '50', '100', '200', '400', '800', '1600'])
        ax.set_title(f'{asset}: {PATHS:,} paths' + (' (log axis)' if asset == 'BTC' else ''))
        ax.set_xlabel('Month')
        ax.set_xlim(0, MONTHS)
    axes[0].set_ylabel('Index (initial = 100)')
    axes[0].legend(loc='upper left')
    fig.suptitle('Simulated price fans, "Financial crisis" scenario, seed 42', x=0.01, y=1.04, ha='left',
                 fontsize=12, fontweight='semibold')
    save(fig, 'monte_carlo_fans.png')


# 4. Terminal outcome ranges per asset ------------------------------------------------------
def fig_terminal(prices):
    s = summarize(prices, MONTHS).set_index('Asset')
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = list(reversed(ASSETS))
    for k, a in enumerate(order):
        c = ASSET_COLOR[a]
        ax.plot([s.loc[a, 'P05 return (%)'], s.loc[a, 'P95 return (%)']], [k, k], color=c, lw=3, alpha=.45,
                solid_capstyle='round')
        ax.plot(s.loc[a, 'Median return (%)'], k, 'o', color=c, ms=9, mec=theme.SURFACE, mew=1.5)
        ax.text(s.loc[a, 'P95 return (%)'] + 3, k,
                f"median {s.loc[a, 'Median return (%)']:+.0f}%   P(loss) {s.loc[a, 'P(loss) (%)']:.0f}%",
                va='center', fontsize=9, color=TEXT_SECONDARY)
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, ls=':')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xscale('symlog', linthresh=50, linscale=1.2)
    ax.set_xticks([-50, -25, 0, 25, 50, 100, 250, 500, 1000, 2500])
    ax.set_xticklabels(['-50', '-25', '0', '25', '50', '100', '250', '500', '1000', '2500'])
    ax.set_xlabel(f'Return over {MONTHS} months, % (bar = 5th to 95th percentile, dot = median; symlog axis)')
    ax.set_title('Terminal outcome distribution by asset, "Financial crisis" scenario')
    ax.set_xlim(left=s['P05 return (%)'].min() - 15, right=s['P95 return (%)'].max() * 12)
    save(fig, 'terminal_outcomes.png')


# 5. Housing stock-flow model --------------------------------------------------------------
def fig_housing_float():
    cfg = HousingFloatConfig()
    hf = simulate_housing_float(cfg)
    alt = simulate_housing_float(replace(cfg, distressed_listing_per_pp_pct=0))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax = axes[0]
    ax.plot(hf.month, hf.home_price_index, color=BLUE, label='With distressed listings')
    ax.plot(alt.month, alt.home_price_index, color=NEUTRAL, ls='--', label='Distressed channel off')
    ax.axhline(100, color=TEXT_SECONDARY, lw=0.8, ls=':')
    ax.set_title('Endogenous home price index')
    ax.set_ylabel('Index (initial = 100)')
    ax.legend(loc='lower left')
    ax = axes[1]
    for col, lab, c, ls, lw in [('active_listings', 'Active listings', BLUE, '-', 2),
                                ('active_buyers', 'Potential buyers', ORANGE, '-', 3.5),
                                ('completed_sales', 'Sales (= buyers while supply suffices)', AQUA, '-', 1.5),
                                ('new_distressed_listings', 'New distressed listings', RED, '-', 2)]:
        ax.plot(hf.month, hf[col] / 1000, color=c, label=lab, ls=ls, lw=lw)
    ax.set_title('Stocks and flows, thousands of units')
    ax.legend(loc='upper left', ncol=1, fontsize=9)
    ax = axes[2]
    ax.plot(hf.month, hf.months_supply, color=BLUE)
    ax.axhline(cfg.target_months_supply, color=TEXT_SECONDARY, lw=1, ls='--')
    ax.text(1, cfg.target_months_supply + .2, 'balanced supply (target)', fontsize=9, color=TEXT_SECONDARY)
    ax.set_title('Months of supply (listings / monthly buyers)')
    for ax in axes:
        ax.set_xlabel('Month')
        ax.set_xlim(0, cfg.months)
    fig.suptitle('Housing stock-flow model, default configuration (1m units, unemployment 4.5% to 10%, lock-in 35%)',
                 x=0.01, y=1.04, ha='left', fontsize=12, fontweight='semibold')
    save(fig, 'housing_float.png')


# 6. Portfolio strategies ------------------------------------------------------------------
def fig_portfolio(prices):
    w = dict(zip(ASSETS, [.25, .10, .15, .10, .10, 0, .30]))
    hold = portfolio(prices, w)
    strat = portfolio(prices, w, 0.30, 0.75, 'SP500')
    x = np.arange(hold.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    for arr, name, c in [(hold, 'Monthly rebalance', NEUTRAL),
                         (strat, 'Drawdown deployment (30% trigger, 75% of cash to SP500)', BLUE)]:
        ax.plot(x, np.median(arr, axis=0) * 100, color=c, label=f'{name}, median')
        ax.plot(x, np.quantile(arr, .05, axis=0) * 100, color=c, ls=':', label=f'{name}, 5th pct')
    ax.axhline(100, color=TEXT_SECONDARY, lw=0.8, ls=':')
    ax.set_title('Portfolio index (initial = 100)')
    ax.set_xlabel('Month')
    ax.legend(loc='upper left', fontsize=8.5)
    ax = axes[1]
    fin_h, fin_s = (hold[:, -1] - 1) * 100, (strat[:, -1] - 1) * 100
    bins = np.linspace(min(fin_h.min(), fin_s.min()), max(fin_h.max(), fin_s.max()), 45)
    ax.hist(fin_h, bins, color=NEUTRAL, alpha=.7, label='Monthly rebalance')
    ax.hist(fin_s, bins, color=BLUE, alpha=.6, label='Drawdown deployment')
    ax.set_title('Distribution of terminal return, %')
    ax.set_xlabel('Return, %')
    ax.legend()
    ax.set_yticks([])
    save(fig, 'portfolio_strategies.png')


# 7. Sensitivity surface: median SP500 return over (unemployment peak, credit peak) --------
def fig_sensitivity():
    u_grid = np.arange(5, 15.1, 1.0)
    c_grid = np.arange(150, 1051, 100)
    z = np.zeros((len(u_grid), len(c_grid)))
    base = SCENARIOS['Financial crisis']
    for i, u in enumerate(u_grid):
        for j, c in enumerate(c_grid):
            _, p = simulate(replace(base, unemployment_peak=u, credit_peak_bps=c), months=MONTHS, paths=300, seed=SEED)
            z[i, j] = np.median(p[:, -1, 0] / p[:, 0, 0] - 1) * 100
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('div', DIVERGING)
    lim = np.abs(z).max()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(z, origin='lower', aspect='auto', cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim),
                   extent=[c_grid[0] - 50, c_grid[-1] + 50, u_grid[0] - .5, u_grid[-1] + .5])
    for i, u in enumerate(u_grid):
        for j, c in enumerate(c_grid):
            ax.text(c, u, f'{z[i, j]:+.0f}', ha='center', va='center', fontsize=7.5,
                    color='white' if abs(z[i, j]) > lim * .55 else theme.TEXT)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=.02)
    cb.set_label('Median SP500 return at month 36, %')
    ax.set_xlabel('Peak high-yield credit spread, bps')
    ax.set_ylabel('Peak unemployment, %')
    ax.set_title('Sensitivity surface: terminal equity return vs. labour and credit stress (300 paths per cell)')
    save(fig, 'sensitivity_surface.png')


# 8. Historical FRED panel ----------------------------------------------------------------
def fig_history():
    from data.historical import fetch_all, CACHE
    if not CACHE.exists() or not list(CACHE.glob('*.csv')):
        print('skip history figure: data/cache is empty')
        return
    got, _ = fetch_all(refresh=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    cs = got['CSUSHPINSA'].pct_change(12) * 100
    ax = axes[0, 0]
    ax.fill_between(cs.index, 0, cs.values, where=cs.values >= 0, color=BLUE, alpha=.6, lw=0)
    ax.fill_between(cs.index, 0, cs.values, where=cs.values < 0, color=RED, alpha=.6, lw=0)
    ax.set_title('Case-Shiller national HPI, year-on-year %')
    ax.axhline(0, color=TEXT_SECONDARY, lw=.8)
    ax = axes[0, 1]
    u = got['UNRATE']
    ax.plot(u.index, u.values, color=ORANGE)
    ax.set_title('Unemployment rate, %')
    ax = axes[1, 0]
    for code, lab, c in [('DGS10', '10Y Treasury', ORANGE), ('MORTGAGE30US', '30Y mortgage', AQUA),
                         ('FEDFUNDS', 'Fed funds', BLUE)]:
        s = got[code].resample('MS').last()
        ax.plot(s.index, s.values, color=c, label=lab, lw=1.5)
    ax.set_title('Policy, Treasury and mortgage rates, %')
    ax.legend(loc='upper right')
    ax.set_xlim(pd.Timestamp('1954-01-01'), None)
    ax = axes[1, 1]
    w = got['WALCL'] / 1e6
    ax.plot(w.index, w.values, color=BLUE)
    ax.set_title('Federal Reserve total assets, $ trillion')
    fig.suptitle('Real FRED observations cached by the Historical tab (no imputation)', x=0.01, ha='left',
                 fontsize=12, fontweight='semibold')
    save(fig, 'historical_fred.png')


if __name__ == '__main__':
    fig_scenarios()
    fig_macro()
    _, prices = simulate(SCENARIOS['Financial crisis'], months=MONTHS, paths=PATHS, seed=SEED)
    fig_fan(prices)
    fig_terminal(prices)
    fig_housing_float()
    fig_portfolio(prices)
    fig_sensitivity()
    fig_history()
