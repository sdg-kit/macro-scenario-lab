"""Download REAL historical data from FRED directly, with local cache.
No bundled dummy series, future interpolation, or synthetic history.
"""
from pathlib import Path
from io import StringIO
import pandas as pd
import requests

SERIES = {
    'UNRATE': ('Unemployment rate', 'Percent', '1948'),
    'CPIAUCSL': ('Consumer price index', 'Index', '1947'),
    'FEDFUNDS': ('Effective Fed funds rate', 'Percent', '1954'),
    'DGS10': ('10Y Treasury yield', 'Percent', '1962'),
    'MORTGAGE30US': ('30Y mortgage rate', 'Percent', '1971'),
    'CSUSHPINSA': ('Case-Shiller national home prices', 'Index', '1987'),
    'USSTHPI': ('FHFA national all-transactions house price index', 'Index', '1975'),
    'BAMLH0A0HYM2': ('High-yield credit spread (ICE BofA; FRED serves only a rolling recent window)', 'Percent', 'rolling'),
    'WALCL': ('Fed total assets', 'Millions USD', '2002'),
    'SP500': ('S&P 500 price index (10-year FRED window)', 'Index', '2016'),
}
CACHE = Path(__file__).resolve().parent / 'cache'

def fetch_one(code: str, refresh: bool=False) -> pd.Series:
    if code not in SERIES:
        raise KeyError(code)
    CACHE.mkdir(exist_ok=True,parents=True)
    target = CACHE / f'{code}.csv'
    if refresh or not target.exists():
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}'
        response = requests.get(url, timeout=25, headers={'User-Agent':'MacroScenarioLab/0.1'})
        response.raise_for_status()
        if code not in response.text[:250]:
            raise RuntimeError(f'FRED returned unexpected content for {code}')
        target.write_bytes(response.content)
    df = pd.read_csv(target)
    date_col = 'observation_date' if 'observation_date' in df else 'DATE'
    dates = pd.to_datetime(df[date_col],errors='coerce')
    vals = pd.to_numeric(df[code],errors='coerce')
    series = pd.Series(vals.to_numpy(), index=dates, name=code).dropna()
    return series[~series.index.isna()].sort_index()

def fetch_all(refresh=False):
    found, problems = {}, {}
    for code in SERIES:
        try:
            found[code] = fetch_one(code, refresh=refresh)
        except Exception as ex:
            problems[code] = str(ex)
    return found, problems

def align_monthly(series: dict) -> pd.DataFrame:
    return pd.concat({k: v.resample('MS').last() for k,v in series.items()},axis=1).sort_index()
