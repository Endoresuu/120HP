# Option Pricer — Quantitative Finance Playground

A Streamlit-based quantitative finance application for pricing, calibration, and hedging of financial derivatives under classical models (Black–Scholes, Heston).

This project is designed as an educational and practical playground for:
- derivative pricing,
- volatility modeling,
- sensitivity analysis (Greeks),
- static replication and dynamic delta hedging,
- interest rate products.

---

## Features

### Derivative Pricing
- European options (Call / Put)
- Black–Scholes pricing
- Implied volatility computation (Newton method)
- Market data integration (option chains)

### Volatility Modeling
- Implied volatility smile (market-calibrated)
- Implied volatility surface (strike × maturity)
- 2D interpolation
- 3D surface visualization

### Greeks Analysis
- Delta, Gamma, Vega, Theta, Rho
- Greeks at a point
- Greeks vs strike
- Gamma heatmap

### Replication & Hedging
- Static replication using Delta
- Payoff comparison vs replication
- Discrete delta hedging simulation (Monte Carlo)
- Hedging error distribution and export

### Stochastic Volatility
- Heston model pricing via Monte Carlo
- Path simulation

### Linear & Rates Products
- Forward contracts
- Futures
- Plain-vanilla fixed-for-floating interest rate swaps
- Par rate and present value computation

---

## Project Structure
## Project Structure

```text
120HP/
├── .vscode/
│   └── settings.json
│
├── interface/
│   ├── .streamlit/
│   │   └── config.toml
│   │
│   ├── tabs/
│   │   ├── tab_pricing.py          # Vanilla option pricing (Black–Scholes)
│   │   ├── tab_smile.py            # Volatility smile calibration
│   │   ├── tab_surface.py          # Volatility surface construction
│   │   ├── tab_heston.py           # Heston stochastic volatility model
│   │   ├── tab_greeks.py           # Greeks analysis (Delta, Gamma, Vega, etc.)
│   │   ├── tab_linear.py           # Linear products (forwards, futures)
│   │   ├── tab_replication.py      # Static replication & delta hedging
│   │   └── tab_swaps.py            # Interest rate swaps
│   │
│   ├── utils/
│   │   └── helpers.py              # UI / formatting helpers
│   │
│   ├── streamlit_app.py            # Main Streamlit entry point
│   └── streamlit_app1.py           # Alternative / dev entry point
│
├── pricer/
│   ├── calibration/
│   │   ├── market_calibrator.py    # Smile & surface calibration logic
│   │   └── surface_calibrator.py
│   │
│   ├── market/
│   │   ├── data.py                 # Market data abstraction
│   │   └── helper.py               # Option chain & expiry helpers
│   │
│   ├── models/
│   │   ├── black_scholes.py        # Black–Scholes model
│   │   ├── heston.py               # Heston model
│   │   └── bs_greeks.py            # Closed-form Greeks
│   │
│   ├── pricing/
│   │   └── engine.py               # Pricing engine
│   │
│   ├── products/
│   │   ├── vanilla.py              # European call & put
│   │   ├── forward.py              # Forward contracts
│   │   ├── future.py               # Futures
│   │   └── swap.py                 # Interest rate swaps
│   │
│   └── __init__.py
│
├── examples/
│   └── notebooks.ipynb             # Optional experimentation
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── picture.png                     # Illustrations / screenshots
```


---

## Installation

### Prerequisites
- Python >= 3.10
- pip

### Setup

```bash

pip install -r requirements.txt
```
### How to run 

```bash

python3 -m streamlit run interface/streamlit_app.py
```
