from pricer.calibration.market_smile_calibrator import MarketSmileCalibrator

def main():
    cal = MarketSmileCalibrator("SPY", r=0.04)

    df = cal.compute_smile()
    print(df)

    cal.plot_smile("smile_SPY.png")


if __name__ == "__main__":
    main()
