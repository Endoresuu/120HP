class MarketData:
    """
    A unified data container for market environment parameters and model calibration results.

    This class stores basic market observables (spot price, interest rates, dividends)
    and serves as a placeholder for stochastic volatility parameters (Heston model)
    and simulation settings.

    Attributes:
        spot (float | None): Current market price of the underlying asset (alias for S_0).
        S_0 (float | None): Initial asset price at time t=0.
        r (float): Annualized risk-free interest rate (default is 0.04 or 4%).
        q (float): Annualized dividend yield of the underlying asset.
        T (float | None): Time to maturity in years.

        # Heston Model Parameters (set during calibration)
        v_0 (float | None): Initial variance of the underlying asset.
        kappa (float | None): Mean-reversion speed of the variance.
        theta (float | None): Long-term mean level of the variance.
        sigma_v (float | None): Volatility of the variance (vol-of-vol).
        rho (float | None): Correlation between the asset price and its variance.

        # Simulation Settings
        n_steps (int | None): Number of time steps for path generation.
        n_paths (int | None): Number of simulated trajectories (Monte Carlo).
    """
    def __init__(self, spot: float | None = None, r: float = 0.04,
                 q: float = 0.0, T: float | None = None):
        """
        Initializes MarketData with essential market observables.

        Args:
            spot (float | None): The current spot price.
            r (float): The risk-free rate. Defaults to 0.04.
            q (float): The dividend yield. Defaults to 0.0.
            T (float | None): Maturity of the options/contracts.
        """
        self.spot = spot
        self.S_0 = spot
        self.r = r
        self.q = q
        self.T = T

        self.v_0 = None
        self.kappa = None
        self.theta = None
        self.sigma_v = None
        self.rho = None
        self.n_steps = None
        self.n_paths = None
