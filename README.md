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

```text
120HP/
├── interface/
│   ├── streamlit_app.py
│   ├── tabs/
│   │   ├── tab_pricing.py
│   │   ├── tab_greeks.py
│   │   ├── tab_replication.py
│   └── utils/
├── pricer/
│   ├── models/
│   ├── products/
│   ├── market/
│   └── pricing/
├── requirements.txt
└── README.md
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
