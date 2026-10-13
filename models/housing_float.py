"""Stylized monthly housing stock/flow model. All coefficients are assumptions.

Counts in housing units; prices are indexes. Vacancy is NOT equivalent to for-sale inventory.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HousingFloatConfig:
    stock: int = 1_000_000
    vacancy_pct: float = 10.0
    initial_inventory_pct: float = 1.2  # active listings / total stock (may be occupied)
    starting_monthly_sales_pct: float = 0.28  # sales / total stock per month
    new_listing_pct: float = 0.32  # new voluntary listings / total stock per month
    withdrawal_pct: float = 5.0  # percent of unsold listings withdrawn per month
    mortgage_lock_in_pct: float = 35.0  # relative reduction in voluntary new listings
    unemployment_start: float = 4.5
    unemployment_peak: float = 10.0
    unemployment_peak_month: int = 18
    distressed_listing_per_pp_pct: float = 0.045  # additional listings per month as % stock / unemployment pp
    distressed_listing_lag: int = 4
    buyer_demand_shock_pct: float = -35.0
    buyer_demand_ramp_months: int = 12
    buyer_rate_sensitivity: float = 0.075  # proportional buyer hit per mortgage pp ABOVE initial
    mortgage_start: float = 6.5
    mortgage_peak: float = 8.0
    mortgage_peak_month: int = 15
    price_sensitivity: float = 0.025  # monthly log change per unit log(inventory/target)
    target_months_supply: float = 5.0
    demand_price_elasticity: float = 0.7  # buyer increase for cheaper homes (log price)
    sale_matching_speed: float = 1.0  # 0-1; fraction of eligible buyers transacting
    months: int = 36


def simulate_housing_float(cfg: HousingFloatConfig) -> pd.DataFrame:
    if cfg.stock <= 0 or cfg.months < 3:
        raise ValueError('stock must be positive and months >= 3')
    if not 0 <= cfg.vacancy_pct <= 100 or not 0 <= cfg.initial_inventory_pct <= 100:
        raise ValueError('percentages must be between 0 and 100')
    if cfg.target_months_supply <= 0 or cfg.price_sensitivity < 0:
        raise ValueError('target supply must be positive and sensitivity nonnegative')
    s = float(cfg.stock)
    inventory = s * cfg.initial_inventory_pct / 100
    initial_inventory = inventory
    initial_sales = s * cfg.starting_monthly_sales_pct / 100
    baseline_voluntary = s * cfg.new_listing_pct / 100
    price = 100.0
    records = []
    unemployment_history = []
    cumulative_sales = 0.0
    cumulative_distressed = 0.0
    for t in range(cfg.months + 1):
        u_scale = min(t / max(1, cfg.unemployment_peak_month), 1)
        unemployment = cfg.unemployment_start + (cfg.unemployment_peak - cfg.unemployment_start)*u_scale
        r_scale = min(t / max(1, cfg.mortgage_peak_month), 1)
        mortgage = cfg.mortgage_start + (cfg.mortgage_peak - cfg.mortgage_start)*r_scale
        unemployment_history.append(unemployment)
        if t == 0:
            voluntary = distressed = sales = withdrawals = buyers = 0.0
            supply_months = initial_inventory / max(initial_sales, 1)
        else:
            lagged = unemployment_history[max(0, t - cfg.distressed_listing_lag)]
            stress_pp = max(0, lagged - cfg.unemployment_start)
            distressed = s * cfg.distressed_listing_per_pp_pct / 100 * stress_pp
            voluntary = baseline_voluntary*(1 - cfg.mortgage_lock_in_pct / 100)
            # Demand responds both to a time-ramped exogenous shock and to affordability.
            shock = (cfg.buyer_demand_shock_pct / 100) * min(t/max(cfg.buyer_demand_ramp_months,1),1)
            rate_effect = -cfg.buyer_rate_sensitivity * (mortgage-cfg.mortgage_start)
            affordability = -cfg.demand_price_elasticity*np.log(max(price,1)/100)
            buyers = initial_sales * max(0.02, 1 + shock + rate_effect + affordability)
            available = max(0, inventory+voluntary+distressed)
            sales = min(available, buyers * np.clip(cfg.sale_matching_speed,0,1))
            remainder = available - sales
            withdrawals = min(remainder, remainder*np.clip(cfg.withdrawal_pct/100,0,1))
            inventory = max(0.0, remainder - withdrawals)
            cumulative_sales += sales
            cumulative_distressed += distressed
            # Baseline inventory / baseline sales sets a reference, but the user can choose
            # another 'balanced' months supply and see the implications.
            realized_months_supply = inventory/max(buyers,1.0)
            imbalance = np.log(max(realized_months_supply,0.05)/cfg.target_months_supply)
            monthly_log_change = -cfg.price_sensitivity*imbalance
            # Bounded per-step to avoid numeric explosions with extreme sliders.
            price *= np.exp(np.clip(monthly_log_change,-0.15,0.15))
            supply_months = realized_months_supply
        records.append(dict(month=t, home_price_index=price, home_price_change_pct=price-100,
            total_homes=s, vacant_homes=s*cfg.vacancy_pct/100, active_listings=inventory,
            effective_float_pct=100*inventory/s, new_voluntary_listings=voluntary,
            new_distressed_listings=distressed, active_buyers=buyers,
            completed_sales=sales, withdrawn_listings=withdrawals,
            months_supply=supply_months, unemployment=unemployment,
            mortgage_rate=mortgage, cumulative_sales=cumulative_sales,
            cumulative_distressed_listings=cumulative_distressed))
    return pd.DataFrame(records)


def diagnostics(frame: pd.DataFrame) -> dict:
    end = frame.iloc[-1]
    return {'final_price_change_pct': float(end.home_price_change_pct),
        'peak_float_pct': float(frame.effective_float_pct.max()),
        'final_months_supply': float(end.months_supply),
        'distressed_listing_share_pct': float(100*frame.new_distressed_listings.sum()/max(1, (frame.new_distressed_listings+frame.new_voluntary_listings).sum()))}
