"""Shared colour system for the dashboard (Plotly) and the README figures (Matplotlib).

Categorical hues are assigned in a fixed order that has been validated for
colour-vision-deficiency separation on both light and dark surfaces.  Entities keep
their colour across every chart: e.g. SP500 is always slot 1, BTC always slot 4.
"""
import plotly.graph_objects as go
import plotly.io as pio

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e6e5e1'

CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL
NEUTRAL = '#9a9892'          # de-emphasis grey for context series
SEQUENTIAL = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
DIVERGING = ['#0d366b', '#2a78d6', '#86b6ef', '#f0efec', '#f0a0a0', '#e34948', '#8c1d1d']

# Entity -> colour.  Fixed so a series never changes hue between tabs or figures.
ASSET_COLOR = {'SP500': BLUE, 'NASDAQ': ORANGE, 'SOXX': AQUA, 'BTC': YELLOW,
               'GOLD': MAGENTA, 'BONDS_10Y': GREEN, 'CASH': VIOLET}
MACRO_COLOR = {'housing': BLUE, 'unemployment': ORANGE, 'inflation': AQUA,
               'fed': BLUE, 'treasury': ORANGE, 'mortgage': AQUA,
               'stimulus': BLUE, 'qe': ORANGE, 'swf': AQUA,
               'credit_bps': RED, 'real_yield': VIOLET}
SCENARIO_COLOR = {'Financial crisis': BLUE, 'High-for-longer': ORANGE,
                  'Sovereign debt crisis': AQUA, 'AI unemployment': YELLOW,
                  'Controlled repricing': MAGENTA}

PRETTY = {'housing': 'House price index', 'unemployment': 'Unemployment', 'inflation': 'Inflation',
          'fed': 'Fed funds', 'treasury': '10Y Treasury', 'mortgage': '30Y mortgage',
          'stimulus': 'Fiscal stimulus', 'qe': 'Fed asset purchases', 'swf': 'Sovereign fund purchases',
          'credit_bps': 'HY credit spread', 'real_yield': 'Real 10Y yield'}

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family='Inter, Segoe UI, Helvetica, Arial, sans-serif', color=TEXT, size=13),
    title=dict(font=dict(size=16, color=TEXT), x=0.01, xanchor='left'),
    colorway=CATEGORICAL,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, showline=False, ticks=''),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, showline=False, ticks=''),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT_SECONDARY)),
    hoverlabel=dict(bgcolor='white', font=dict(color=TEXT)),
    hovermode='x unified',
    margin=dict(l=40, r=20, t=55, b=40),
)
_template.data.scatter = [go.Scatter(line=dict(width=2))]
pio.templates['lab'] = _template
pio.templates.default = 'lab'


def mpl_style():
    """Apply the same system to Matplotlib (used by scripts/make_figures.py)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_SECONDARY, 'axes.titlecolor': TEXT,
        'axes.titleweight': 'semibold', 'axes.titlesize': 12, 'axes.titlelocation': 'left',
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'text.color': TEXT, 'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'lines.linewidth': 2.0, 'legend.frameon': False, 'legend.fontsize': 10,
        'axes.prop_cycle': mpl.cycler(color=CATEGORICAL),
        'figure.dpi': 150, 'savefig.dpi': 170, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.25,
    })


def hex_alpha(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)' for Plotly fills."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'
