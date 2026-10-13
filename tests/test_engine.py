import numpy as np
import pytest
from models.engine import Scenario, Baseline, Sensitivities, simulate, portfolio, ASSETS

def test_housing_trough_and_finite():
    s=Scenario(bottom_month=18,housing_drop=40)
    macro,p=simulate(s,months=36,paths=50)
    assert abs(macro.loc[18,'housing']-60)<1e-8
    assert p.shape==(50,37,len(ASSETS))
    assert (p>0).all() and np.isfinite(p).all()

def test_determinism():
    _,x=simulate(Scenario(),months=24,paths=10,seed=1)
    _,y=simulate(Scenario(),months=24,paths=10,seed=1)
    assert np.array_equal(x,y)

def test_portfolio_weights():
    _,p=simulate(Scenario(),months=12,paths=25)
    with pytest.raises(ValueError):
        portfolio(p,{'SP500':.5})
    a=portfolio(p,{'CASH':1.0})
    assert a.shape==(25,13)
    assert (np.diff(a,axis=1)>=0).all()

def test_deployment_only_when_hit():
    _,p=simulate(Scenario(),months=18,paths=20)
    w={'SP500':.5,'CASH':.5}
    assert np.allclose(portfolio(p,w),portfolio(p,w,.99,1.0))
