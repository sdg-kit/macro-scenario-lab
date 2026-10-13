import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import replace as _replace
import theme
from theme import ASSET_COLOR, MACRO_COLOR, PRETTY, NEUTRAL, TEXT_SECONDARY
from models.engine import (ASSETS, DEFAULT_PRICES, Scenario, SCENARIOS,
                           Baseline, Sensitivities, simulate, summarize, portfolio)
from data.historical import SERIES, fetch_all, align_monthly
from models.housing_float import HousingFloatConfig, simulate_housing_float, diagnostics

st.set_page_config(page_title='Macro Scenario Lab', page_icon='📉', layout='wide')
st.title('Macro Scenario Lab')
st.caption('Housing shocks → macro paths → asset responses → portfolio outcomes | Transparent, assumption-driven exploratory model')

with st.sidebar:
    st.header('Scenario controls')
    picked = st.selectbox('Starting scenario',list(SCENARIOS))
    sc0 = SCENARIOS[picked]
    horizon = st.slider('Simulation horizon (months)',12,72,36,6)
    housing_drop = st.slider('Housing price decline (%)',0,60,int(sc0.housing_drop))
    bottom = st.slider('Housing trough month',3,horizon,min(sc0.bottom_month,horizon))
    peak_u = st.slider('Peak unemployment (%)',3.0,18.0,float(sc0.unemployment_peak),.5)
    inflation = st.slider('End annual inflation (%)',-3.0,15.0,float(sc0.inflation_end),.5)
    fed = st.slider('End Fed funds (%)',0.0,12.0,float(sc0.fed_end),.25)
    ty = st.slider('End 10Y yield (%)',0.0,12.0,float(sc0.treasury_end),.25)
    credit = st.slider('Peak HY credit spread (bps)',100,1200,int(sc0.credit_peak_bps),25)
    st.subheader('Government and productivity')
    stimulus = st.slider('Fiscal stimulus ($ trillion)',0.0,5.0,float(sc0.stimulus_trillions),.1)
    qe = st.slider('Fed asset purchases ($ trillion)',0.0,5.0,float(sc0.qe_trillions),.1)
    swf = st.slider('Sovereign fund equity purchases ($ trillion)',0.0,2.0,float(sc0.sovereign_fund_trillions),.1)
    policy = st.slider('Policy intervention starts, month',0,horizon,min(sc0.policy_start_month,horizon))
    ai = st.slider('AI productivity proxy (%)',0.0,15.0,float(sc0.ai_productivity),.5)
    st.subheader('Simulation')
    count = st.select_slider('Monte Carlo paths', options=[100,250,500,1000,2500,5000], value=1000)
    seed = st.number_input('Random seed',0,999999,42)
    with st.expander('Starting macro levels'):
        bu = st.number_input('Unemployment (%)',0.,20.,4.5,.1)
        bi = st.number_input('Inflation (%)',-5.,20.,3.,.1)
        bf = st.number_input('Fed funds (%)',0.,15.,4.,.1)
        by = st.number_input('10Y (%)',0.,15.,4.5,.1)
        bm = st.number_input('Mortgage (%)',0.,20.,6.5,.1)
        bc = st.number_input('HY credit spread (bps)',0.,2500.,130.,10.)
    with st.expander('Asset model coefficients'):
        eqc = st.slider('Equity sensitivity to credit widening',0.,.20,.065,.005)
        equ = st.slider('Equity sensitivity to unemployment',0.,.20,.075,.005)
        eqy = st.slider('Equity sensitivity to yield changes',0.,.20,.045,.005)
        eqs = st.slider('Equity boost per $T stimulus',0.,.12,.016,.002)
        eqq = st.slider('Equity boost per $T QE',0.,.12,.022,.002)
        eqw = st.slider('Equity boost per $T sovereign purchases',0.,.20,.055,.005)
        btcl = st.slider('BTC sensitivity to liquidity',0.,.6,.22,.01)
        btcc = st.slider('BTC sensitivity to credit',0.,.4,.14,.01)
        btcy = st.slider('BTC sensitivity to real yields',0.,.3,.1,.01)
        gi = st.slider('Gold sensitivity to inflation',0.,.15,.035,.005)
        gr = st.slider('Gold sensitivity to real yields',0.,.2,.06,.005)
        duration = st.slider('10Y bond duration',1.,15.,7.,.5)
        volmult = st.slider('Asset shock volatility multiplier',0.,2.5,1.,.1)

sc = Scenario(picked,housing_drop,peak_u,inflation,fed,ty,credit,stimulus,qe,swf,ai,policy,bottom)
base = Baseline(bu,bi,bf,by,bm,bc)
sens = Sensitivities(eqc,equ,eqy,eqs,eqq,eqw,btcl,btcc,btcy,gi,gr,duration,volmult)
macro, prices = simulate(sc,base,sens,horizon,count,int(seed))


def lines(df, cols, title, ytitle='', baseline=None):
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df['month'],y=df[c],name=PRETTY.get(c,c),mode='lines',
                                 line=dict(color=MACRO_COLOR.get(c))))
    fig.update_layout(title=title,height=370,margin=dict(l=30,r=20,b=30,t=55),
                      xaxis_title='Month',yaxis_title=ytitle,legend=dict(orientation='h',y=-.24))
    return fig


def uncertainty(a, asset, label):
    idx=ASSETS.index(asset)
    normalized=a[:,:,idx]/a[:,0:1,idx]*100
    x=np.arange(a.shape[1]); lo, med, hi = np.quantile(normalized,[.05,.5,.95],axis=0)
    c=ASSET_COLOR[asset]
    q25,q75=np.quantile(normalized,[.25,.75],axis=0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x,y=hi,line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x,y=lo,fill='tonexty',line=dict(width=0),name='5th–95th percentile',fillcolor=theme.hex_alpha(c,.18)))
    fig.add_trace(go.Scatter(x=x,y=q75,line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x,y=q25,fill='tonexty',line=dict(width=0),name='25th–75th percentile',fillcolor=theme.hex_alpha(c,.32)))
    fig.add_trace(go.Scatter(x=x,y=med,name='Median',line=dict(width=3,color=c)))
    fig.add_hline(y=100,line_dash='dot',line_color=TEXT_SECONDARY)
    fig.update_layout(title=label+' (initial = 100)',height=360,xaxis_title='Month',yaxis_title='Index',
       margin=dict(l=30,r=20,b=30,t=55))
    return fig

names = ['Scenario dashboard','Housing float','Monte Carlo','Portfolio strategies','Historical data','Model notes']
t1,tfloat,t2,t3,t4,t5 = st.tabs(names)
with t1:
    st.info('All returns are simulated conditional scenarios, not empirical forecasts. Macro starting levels are editable assumptions, not automatically live market quotes.')
    a,b,c,d = st.columns(4)
    a.metric('Housing change at trough',f'{macro.loc[bottom,"housing"]-100:.1f}%')
    b.metric('Median SP500 return at horizon',f'{np.median(prices[:,-1,0]/prices[:,0,0]-1)*100:+.1f}%')
    c.metric('Median BTC return at horizon',f'{np.median(prices[:,-1,3]/prices[:,0,3]-1)*100:+.1f}%')
    d.metric('10Y yield at horizon',f'{macro.iloc[-1].treasury:.2f}%')
    x,y=st.columns(2)
    with x:
        st.plotly_chart(lines(macro,['housing'],'House prices (initial = 100)'),use_container_width=True)
        st.plotly_chart(lines(macro,['fed','treasury','mortgage'],'Rates','percent'),use_container_width=True)
    with y:
        st.plotly_chart(lines(macro,['unemployment','inflation'],'Labor and inflation','percent'),use_container_width=True)
        st.plotly_chart(lines(macro,['stimulus','qe','swf'],'Cumulative policy programs','$ trillion'),use_container_width=True)
    st.subheader('Terminal asset outcomes')
    summary=summarize(prices)
    st.dataframe(summary.style.format({col:'{:+,.1f}' for col in summary.columns if '(%)' in col}),use_container_width=True,hide_index=True)
    st.download_button('Download macro paths CSV',macro.to_csv(index=False),'macro_paths.csv','text/csv')
with tfloat:
    st.subheader('Housing float and price discovery')
    st.caption('Exploratory monthly stock-flow system. Vacancy is separate from listings. This endogenous housing trajectory is independent of the original asset-return scenarios.')
    lc, rc, ec = st.columns(3)
    with lc:
        stock_m = st.slider('Housing stock (millions)', 0.1, 150.0, 1.0, .1, key='hf_stock')
        vacancy = st.slider('Vacant homes (% of stock)', 0.0, 30.0, 10.0, .5)
        start_inventory = st.slider('Initially listed (% of stock)', 0.1, 6.0, 1.2, .1)
        start_sales = st.slider('Monthly sales (% of stock)', 0.05, 1.0, .28, .01)
        voluntary = st.slider('New voluntary listings (% of stock/month)', 0., 1., .32, .01)
        withdrawal = st.slider('Unsold listing withdrawals (%/month)', 0., 30., 5., 1.)
    with rc:
        lock = st.slider('Mortgage lock-in: voluntary-listing reduction (%)', 0., 90., 35., 5.)
        peak_unemployment = st.slider('Peak unemployment (%)', 3., 18., 10., .5, key='hf_unemp')
        distressed = st.slider('Distressed listings (% stock per unemployment pp/month)', 0., .20, .045, .005)
        lag = st.slider('Distressed listing lag (months)', 0, 18, 4)
        buyer_shock = st.slider('Buyer demand shock (%)', -90., 80., -35., 5.)
        mortgage_peak = st.slider('Peak mortgage rate (%)', 2., 15., 8., .25)
    with ec:
        sensitivity = st.slider('Monthly price response speed', .0, .12, .025, .005)
        target = st.slider('Balanced months supply', 1., 12., 5., .5)
        elasticity = st.slider('Buyer demand response to lower prices', 0., 3., .7, .1)
        matching = st.slider('Buyer matching efficiency', .1, 1., 1., .05)
        hf_months = st.slider('Housing simulation horizon (months)', 12, 96, 36, 6)
        compare = st.checkbox('Compare no-distressed-listings scenario', True)
    cfg = HousingFloatConfig(stock=round(stock_m*1_000_000), vacancy_pct=vacancy,
        initial_inventory_pct=start_inventory, starting_monthly_sales_pct=start_sales,
        new_listing_pct=voluntary, withdrawal_pct=withdrawal,
        mortgage_lock_in_pct=lock, unemployment_peak=peak_unemployment,
        distressed_listing_per_pp_pct=distressed, distressed_listing_lag=lag,
        buyer_demand_shock_pct=buyer_shock, mortgage_peak=mortgage_peak,
        price_sensitivity=sensitivity, target_months_supply=target,
        demand_price_elasticity=elasticity, sale_matching_speed=matching, months=hf_months)
    hf = simulate_housing_float(cfg)
    dg = diagnostics(hf)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric('Price change', f'{dg["final_price_change_pct"]:+.1f}%')
    k2.metric('Peak listed float', f'{dg["peak_float_pct"]:.2f}%')
    k3.metric('Final months supply', f'{dg["final_months_supply"]:.1f}')
    k4.metric('Distressed share of new listings', f'{dg["distressed_listing_share_pct"]:.1f}%')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hf.month,y=hf.home_price_index,name='Price index',line=dict(width=3,color=theme.BLUE)))
    if compare:
        alt = simulate_housing_float(_replace(cfg, distressed_listing_per_pp_pct=0))
        fig.add_trace(go.Scatter(x=alt.month,y=alt.home_price_index,name='No distressed listings',line=dict(dash='dash',color=NEUTRAL)))
    fig.add_hline(y=100,line_dash='dot',line_color=TEXT_SECONDARY)
    fig.update_layout(title='Endogenous home prices (initial = 100)',height=380,xaxis_title='Month',yaxis_title='Index')
    st.plotly_chart(fig,use_container_width=True)
    fig2 = go.Figure()
    for label, column, colour in [('Listings','active_listings',theme.BLUE),('Potential buyers','active_buyers',theme.ORANGE),('Sales','completed_sales',theme.AQUA),('New distressed listings','new_distressed_listings',theme.RED)]:
        fig2.add_trace(go.Scatter(x=hf.month,y=hf[column],name=label,line=dict(color=colour)))
    fig2.update_layout(title='Housing participation and transaction flow',height=370,xaxis_title='Month',yaxis_title='Units')
    a,b = st.columns(2)
    with a: st.plotly_chart(fig2,use_container_width=True)
    with b:
        fig3 = go.Figure(go.Scatter(x=hf.month,y=hf.months_supply,name='Months supply',line=dict(color=theme.BLUE)))
        fig3.add_hline(y=target,line_dash='dash',line_color=TEXT_SECONDARY,annotation_text='Balanced supply')
        fig3.update_layout(title='Inventory / current monthly buyer absorption',height=370,xaxis_title='Month',yaxis_title='Months')
        st.plotly_chart(fig3,use_container_width=True)
    st.download_button('Export housing simulation CSV',hf.to_csv(index=False),'housing_float_paths.csv','text/csv')
    with st.expander('Definitions and limitations'):
        st.write('Vacancy is not the same as active listing inventory, and occupied properties can be listed. Listings change through newly offered properties, sales, and withdrawals. Mortgage lock-in reduces voluntary listings; unemployment raises distressed listings after a lag. Price adjusts to months of supply, using a user-chosen coefficient. This is an illustrative mechanism, not a historically estimated price elasticity. Buyers are monthly potential purchases rather than a census of distinct buyers. The housing simulation is not yet fed into the other tabs asset-pricing model.')

with t2:
    st.subheader('Asset uncertainty ranges')
    c1,c2=st.columns([3,1])
    sel=c1.selectbox('Asset to inspect',ASSETS,index=3)
    logy=c2.checkbox('Log scale',value=(sel=='BTC'),help='Recommended for heavy-tailed assets so the median is not flattened by the upper tail.')
    fan=uncertainty(prices,sel,sel)
    if logy: fan.update_yaxes(type='log',tickvals=[12.5,25,50,100,200,400,800,1600,3200,6400],ticktext=['12.5','25','50','100','200','400','800','1600','3200','6400'])
    st.plotly_chart(fan,use_container_width=True)
    st.write('Quantiles reflect simulated return shocks and user-set economic conditions. They are not confidence intervals fitted to history.')
    st.subheader('Probability of an illustrative threshold')
    threshold=st.slider('Percent below initial value',5,90,40,5)
    j=ASSETS.index(sel)
    hit=(prices[:,:,j].min(axis=1)/prices[:,0,j] <= 1-threshold/100).mean()
    st.metric(f'Simulated share ever dropping ≥{threshold}%',f'{100*hit:.1f}%')
    summary_mc=summarize(prices)
    st.dataframe(summary_mc.style.format({col:'{:+,.1f}' for col in summary_mc.columns if '(%)' in col}),use_container_width=True,hide_index=True)
with t3:
    st.subheader('Model portfolio')
    cols=st.columns(4)
    defaults=[25,10,15,10,10,0,30]
    weights={}
    for k,asset in enumerate(ASSETS):
        with cols[k%4]:
            weights[asset]=st.number_input(asset+' weight (%)',0,100,defaults[k],1)/100
    total=sum(weights.values())
    if abs(total-1)>0.00001:
        st.error(f'Weights sum to {total*100:.0f}%. Adjust to exactly 100%.')
    else:
        threshold=st.slider('Market drawdown trigger (%)',10,70,30,5)
        fraction=st.slider('Deploy share of remaining cash at trigger (%)',0,100,75,5)/100
        target=st.selectbox('Buy asset upon trigger',ASSETS[:-1])
        hold=portfolio(prices,weights)
        strategy=portfolio(prices,weights,threshold/100,fraction,target)
        fig=go.Figure()
        for arr,name,colour in [(hold,'Rebalance monthly',NEUTRAL),(strategy,'Drawdown deployment',theme.BLUE)]:
            for q,dash in [(.5,'solid'),(.05,'dot')]:
                fig.add_trace(go.Scatter(x=np.arange(horizon+1),y=np.quantile(arr,q,axis=0)*100,
                    name=name+(' median' if q==.5 else ' 5th percentile'),line=dict(dash=dash,color=colour)))
        fig.add_hline(y=100,line_dash='dot',line_color=TEXT_SECONDARY)
        fig.update_layout(title='Portfolio index (initial = 100)',xaxis_title='Month',height=480)
        st.plotly_chart(fig,use_container_width=True)
        outcomes=pd.DataFrame({'Strategy':['Monthly rebalance','Drawdown deployment'],
          'Median final return %':[100*(np.median(v[:,-1])-1) for v in (hold,strategy)],
          'P05 final return %':[100*(np.quantile(v[:,-1],.05)-1) for v in (hold,strategy)],
          'Median max drawdown %':[100*np.median((v/np.maximum.accumulate(v,axis=1)-1).min(axis=1)) for v in (hold,strategy)]})
        st.dataframe(outcomes,hide_index=True,use_container_width=True)
        st.caption('Deployment triggers at month-end, affects subsequent monthly returns. No transaction costs or taxes; initial weights rebalance monthly until deployment.')
with t4:
    st.subheader('Historical economic research: real FRED observations')
    st.markdown('Data downloaded directly from [FRED](https://fred.stlouisfed.org/); first download needs internet. Cached source CSVs are retained locally, with no imputed dates or fabricated observations.')
    st.dataframe(pd.DataFrame([{'Series':k,'Description':v[0],'Units':v[1],'Approx. first year':v[2]} for k,v in SERIES.items()]),hide_index=True,use_container_width=True)
    if st.button('Download / refresh real FRED series',type='primary'):
        with st.spinner('Retrieving source observations'):
            got, problems = fetch_all(refresh=True)
            st.session_state['hist_got']=got
            st.session_state['hist_problems']=problems
    elif 'hist_got' not in st.session_state:
        # Local cache works without a network connection.
        from data.historical import CACHE
        if CACHE.exists() and list(CACHE.glob('*.csv')):
            got, problems=fetch_all(refresh=False)
            st.session_state['hist_got']=got
            st.session_state['hist_problems']=problems
    if 'hist_got' in st.session_state:
        got=st.session_state['hist_got']; problems=st.session_state['hist_problems']
        if got:
            s=st.selectbox('Historical series',list(got))
            v=got[s]
            st.success(f'{s}: {len(v):,} actual observations, {v.index.min():%Y-%m-%d} through {v.index.max():%Y-%m-%d}')
            fig=go.Figure(go.Scatter(x=v.index,y=v.values,name=s,line=dict(color=theme.BLUE)))
            fig.update_layout(height=430,title=SERIES[s][0],xaxis_title='Date',yaxis_title=SERIES[s][1])
            st.plotly_chart(fig,use_container_width=True)
            monthly=align_monthly(got)
            st.download_button('Download merged monthly history CSV',monthly.to_csv().encode(),
                'fred_monthly_history.csv','text/csv')
        if problems:
            with st.expander(f'{len(problems)} series unavailable (connection or source issue)'):
                st.json(problems)
    else:
        st.caption('No cached history yet; select download to retrieve real data.')
    st.markdown('For pre-1975 US housing and a long-run stock series dating to 1871, see [Robert Shiller’s data](https://www.econ.yale.edu/~shiller/data.htm). Those are different historical constructions and are not automatically merged into FRED series.')
    st.warning('1950s coverage exists for some macro series, not for every asset: Bitcoin started in 2009; mortgage surveys, modern housing indexes, and FRED SP500 data start later.')
with t5:
    st.subheader('Interpretation and caveats')
    st.markdown('''**This is a sensitivity-testing engine, not a historical-calibrated forecasting system.** The housing decline is an input, not an output predicted from Treasury yields. Unemployment, credit spreads and rate paths are also direct scenario assumptions. Asset responses are model equations with editable sensitivities plus stochastic shocks. They do not identify causal effects.\n\n**Sovereign wealth fund:** purchases are modeled as direct additional equity demand; whether a fund exists, how it is financed, and its actual allocations remain separate policy questions. The effect coefficient is a modeling choice rather than a measured policy multiplier.\n\n**Bitcoin and gold:** neither has a complete comparable market-price history back to the 1950s. Asset shock volatility and correlations are stylized.\n\n**Useful next upgrade:** historical rolling calibration of coefficients (no lookahead), regime-conditioned residual bootstrap, empirical recession windows, and a transparent financing constraint on stimulus and sovereign purchases.''')
    st.code('streamlit run app.py',language='bash')
