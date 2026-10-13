from dataclasses import replace
from models.housing_float import HousingFloatConfig, simulate_housing_float


def test_no_negative_stock_flows():
    df = simulate_housing_float(HousingFloatConfig())
    assert len(df) == 37
    assert (df[['active_listings', 'completed_sales', 'withdrawn_listings', 'new_distressed_listings']] >= 0).all().all()
    assert (df.home_price_index > 0).all()


def test_more_demand_raises_final_prices():
    base = HousingFloatConfig()
    stressed = simulate_housing_float(replace(base, buyer_demand_shock_pct=-65))
    strong = simulate_housing_float(replace(base, buyer_demand_shock_pct=20))
    assert strong.home_price_index.iloc[-1] > stressed.home_price_index.iloc[-1]


def test_distressed_supply_lowers_final_prices():
    base = HousingFloatConfig()
    none = simulate_housing_float(replace(base, distressed_listing_per_pp_pct=0))
    extra = simulate_housing_float(replace(base, distressed_listing_per_pp_pct=0.12))
    assert extra.home_price_index.iloc[-1] < none.home_price_index.iloc[-1]


def test_vacancy_not_automatically_listed():
    base = HousingFloatConfig()
    low = simulate_housing_float(replace(base, vacancy_pct=5))
    high = simulate_housing_float(replace(base, vacancy_pct=25))
    assert low.active_listings.equals(high.active_listings)
    assert high.vacant_homes.iloc[0] > low.vacant_homes.iloc[0]


def test_reproducible():
    cfg = HousingFloatConfig()
    assert simulate_housing_float(cfg).equals(simulate_housing_float(cfg))
