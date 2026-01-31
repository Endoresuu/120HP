import os
import sys
import streamlit as st

# ======================================================
# 1) PYTHON PATH
# ======================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ======================================================
# 2) PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Option Pricer",
    layout="wide"
)

# ======================================================
# 3) IMPORT DES TABS
# ======================================================
from tabs.tab_pricing import render as render_pricer
from tabs.tab_smile import render as render_smile
from tabs.tab_surface import render as render_surface
from tabs.tab_heston import render as render_heston
from tabs.tab_greeks import render as render_greeks
from tabs.tab_linear import render as render_linear
from tabs.tab_replication import render as render_replication
from tabs.tab_swaps import render as render_swaps

# ======================================================
# 4) HEADER PRINCIPAL
# ======================================================
st.title("Option Pricer")
st.caption("Pricing, volatility structures and hedging analysis")

# ======================================================
# 5) TABS
# ======================================================
tabs = st.tabs([
    "Pricing",
    "Volatility Smile",
    "Volatility Surface",
    "Heston Model",
    "Greeks",
    "Linear Products",
    "Replication",
    "Swaps"
])

with tabs[0]:
    render_pricer()

with tabs[1]:
    render_smile()

with tabs[2]:
    render_surface()

with tabs[3]:
    render_heston()

with tabs[4]:
    render_greeks()

with tabs[5]:
    render_linear()

with tabs[6]:
    render_replication()

with tabs[7]:
    render_swaps()
