from pricer.calibration.market_calibrator import MarketSmileCalibrator

cal = MarketSmileCalibrator("SPY", r=0.04)

df = cal.compute_smile()
print(df)

cal.plot_smile("smile_SPY.png")
