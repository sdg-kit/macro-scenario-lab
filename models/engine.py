"""Transparent illustrative macro scenario simulator; NOT fitted to historical data."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

ASSETS = ['SP500', 'NASDAQ', 'SOXX', 'BTC', 'GOLD', 'BONDS_10Y', 'CASH']
DEFAULT_PRICES = dict(SP500=100.0, NASDAQ=100.0, SOXX=100.0, BTC=84000.0, GOLD=100.0, BONDS_10Y=100.0, CASH=100.0)

@dataclass
class Scenario:
    name: str = 'Financial crisis'
    housing_drop: float = 40.0
    unemployment_peak: float = 10.0
    inflation_end: float = 1.0
    fed_end: float = 0.5
    treasury_end: float = 2.5
    credit_peak_bps: float = 550.0
    stimulus_trillions: float = 1.5
    qe_trillions: float = 2.0
    sovereign_fund_trillions: float = 0.0
    ai_productivity: float = 2.0
    policy_start_month: int = 9
    bottom_month: int = 24

SCENARIOS = {
    'Financial crisis': Scenario(),
    'High-for-longer': Scenario('High-for-longer', 35, 7, 4.5, 5, 6, 280, 0.5, 0, 0, 2, 14, 30),
    'Sovereign debt crisis': Scenario('Sovereign debt crisis', 25, 8, 9, 7, 8, 450, 2, 3, 0.5, 2, 12, 32),
    'AI unemployment': Scenario('AI unemployment', 40, 12, 0.5, 0.75, 2.5, 360, 1, 1.5, 0.5, 9, 10, 30),
    'Controlled repricing': Scenario('Controlled repricing', 30, 5, 2, 3, 3.75, 160, 0.3, 0, 0, 3, 18, 36),
}

@dataclass
class Baseline:
    unemployment: float = 4.5
    inflation: float = 3.0
    fed: float = 4.0
    treasury: float = 4.5
    mortgage: float = 6.5
    credit_bps: float = 130

@dataclass
class Sensitivities:
    equity_credit: float = 0.065     # monthly log return per 100 bps monthly widening
    equity_unemployment: float = 0.075 # monthly log return per 1pp increase
    equity_yield: float = 0.045
    equity_stimulus: float = 0.016    # log return per trillion new stimulus in a month
    equity_qe: float = 0.022
    equity_swf: float = 0.055
    btc_liquidity: float = 0.22
    btc_credit: float = 0.14
    btc_yield: float = 0.10
    gold_inflation: float = 0.035
    gold_real_yield: float = 0.060
    bond_duration: float = 7.0
    risk_vol_mult: float = 1.0


def _ease(x):
    x = np.clip(x, 0, 1)
    return 3*x*x - 2*x*x*x


def macro_path(sc: Scenario, months: int = 36, base: Baseline = Baseline()) -> pd.DataFrame:
    """Paths are conditional user assumptions; they are not causal macro forecasts."""
    t = np.arange(months + 1, dtype=float)
    bottom = int(np.clip(sc.bottom_month, 3, months))
    d = _ease(t / bottom)
    h = 100 * (1 - sc.housing_drop / 100 * d)
    # A modest bounce after the prescribed housing trough.
    after = np.clip((t - bottom) / max(1, months-bottom), 0, 1)
    h *= (1 + .035 * after)
    hump = np.sin(np.pi * np.clip(t / bottom, 0, 1) / 2)**2
    hump *= np.where(t > bottom, np.exp(-(t-bottom)/14), 1)
    long = _ease(t / months)
    u = base.unemployment + (sc.unemployment_peak-base.unemployment)*hump
    infl = base.inflation + (sc.inflation_end-base.inflation)*long
    fed = base.fed + (sc.fed_end-base.fed)*_ease(t/max(6,sc.policy_start_month+8))
    treasury = base.treasury + (sc.treasury_end-base.treasury)*long
    credit = base.credit_bps + (sc.credit_peak_bps-base.credit_bps)*hump
    mortgage = np.maximum(0, treasury + (base.mortgage-base.treasury) + .0015*(credit-base.credit_bps))
    # Cumulative policy money, not instant simultaneous injections.
    rollout = _ease((t-sc.policy_start_month)/12)
    out = pd.DataFrame({'month': t.astype(int), 'housing': h, 'unemployment': u, 'inflation': infl,
      'fed': fed, 'treasury': treasury, 'credit_bps': credit, 'mortgage': mortgage,
      'stimulus': sc.stimulus_trillions*rollout, 'qe': sc.qe_trillions*rollout,
      'swf': sc.sovereign_fund_trillions*rollout,
      'ai_productivity': np.full(len(t), sc.ai_productivity)})
    out['real_yield'] = out['treasury'] - out['inflation']
    return out


def simulate(sc: Scenario, base: Baseline = Baseline(), sens: Sensitivities = Sensitivities(),
             months: int = 36, paths: int = 1000, seed: int = 42,
             start_prices: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns macro DataFrame and price tensor (paths x months+1 x assets)."""
    if paths < 1 or months < 3:
        raise ValueError('paths must be >=1 and months >=3')
    df = macro_path(sc, months, base)
    rng = np.random.default_rng(seed)
    n = len(ASSETS)
    prices = np.ones((paths, months+1, n))
    initial = {**DEFAULT_PRICES, **(start_prices or {})}
    prices[:,0,:] = [initial[a] for a in ASSETS]
    # Monthly return innovations share a risk factor, allowing joint crashes.
    vol = np.array([.045, .060, .082, .17, .04, .012, 0]) * sens.risk_vol_mult
    beta = np.array([.8, 1.05, 1.30, 1.55, -.12, -.20, 0])
    for i in range(1, months+1):
        prev, cur = df.iloc[i-1], df.iloc[i]
        du = cur.unemployment - prev.unemployment
        dc = (cur.credit_bps-prev.credit_bps)/100
        dy = cur.treasury-prev.treasury
        dinfl = cur.inflation-prev.inflation
        dr = cur.real_yield-prev.real_yield
        ds, dq, dw = [cur[k]-prev[k] for k in ('stimulus','qe','swf')]
        equity = (.004 - sens.equity_credit*dc - sens.equity_unemployment*du
                  - sens.equity_yield*dy + sens.equity_stimulus*ds
                  + sens.equity_qe*dq + sens.equity_swf*dw)
        liq = ds + dq + dw
        btc = .007 + sens.btc_liquidity*liq - sens.btc_credit*dc - sens.btc_yield*dr - .085*du
        gold = .003 + sens.gold_inflation*dinfl - sens.gold_real_yield*dr + .015*dq + .008*dc
        bond = (prev.treasury/100/12) - sens.bond_duration*dy/100
        cash = max(prev.fed,0)/100/12
        expected = np.array([equity, 1.25*equity + .002*sc.ai_productivity/12,
           1.6*equity + .004*sc.ai_productivity/12, btc, gold, bond, cash])
        common = rng.normal(size=(paths,1))
        idio = rng.normal(size=(paths,n))
        shocks = (common*beta + idio*np.sqrt(np.maximum(0,1-beta**2/3)))*vol
        shocks[:,-1] = 0
        log_returns = expected + shocks
        # Bond model uses simple monthly return approximation; clip pathological losses.
        log_returns[:,5] = np.maximum(log_returns[:,5], -0.95)
        prices[:,i,:] = prices[:,i-1,:]*np.exp(log_returns)
    return df, prices


def summarize(prices: np.ndarray, months: int = 36) -> pd.DataFrame:
    perf = prices[:,-1,:]/prices[:,0,:]-1
    wealth = prices/prices[:,0:1,:]
    dd = wealth/np.maximum.accumulate(wealth, axis=1)-1
    return pd.DataFrame({'Asset': ASSETS,
      'Median return (%)': np.median(perf,axis=0)*100,
      'P05 return (%)': np.quantile(perf,.05,axis=0)*100,
      'P95 return (%)': np.quantile(perf,.95,axis=0)*100,
      'Median max drawdown (%)': np.median(dd.min(axis=1),axis=0)*100,
      'P(loss) (%)': (perf<0).mean(axis=0)*100})


def portfolio(prices: np.ndarray, weights: dict, buy_threshold: float | None = None,
              deploy_fraction: float = 0.0, target: str = 'SP500') -> np.ndarray:
    """Monthly-rebalanced portfolio, or threshold cash deployment into target; no leverage.
    If threshold is set, deployment executes at month-end when prior peak drawdown crosses it.
    Returns index (paths x periods). Execution pricing friction omitted.
    """
    w = np.array([max(0.0,float(weights.get(a,0))) for a in ASSETS])
    if not np.isclose(w.sum(),1,atol=1e-5):
        raise ValueError('Portfolio weights must sum to 100%')
    out = np.ones(prices.shape[:2]); active = np.tile(w,(prices.shape[0],1))
    triggered = np.zeros(prices.shape[0],dtype=bool)
    j = ASSETS.index(target); cash = ASSETS.index('CASH')
    for m in range(1,prices.shape[1]):
        rets = prices[:,m,:]/prices[:,m-1,:]
        out[:,m] = out[:,m-1] * (active*rets).sum(axis=1)
        if buy_threshold is not None:
            prior_high = prices[:,:m+1,j].max(axis=1)
            drop = prices[:,m,j]/prior_high-1
            mask = (~triggered) & (drop <= -buy_threshold)
            shifted = active[mask,cash]*deploy_fraction
            active[mask,cash] -= shifted
            active[mask,j] += shifted
            triggered |= mask
    return out
