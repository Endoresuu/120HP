import numpy as np

class InterestRateSwap:
    """
    Representation of a Vanilla Interest Rate Swap (IRS).

    In an IRS, two parties exchange interest rate cash flows: one based on a
    fixed rate and the other on a floating rate (typically a reference index
    like LIBOR or EURIBOR).

    Attributes:
        fixed_rate (float): The annualized fixed rate agreed in the contract.
        T (float): Time to maturity in years.
        freq (int): Number of payments per year (e.g., 1 for annual, 2 for semi-annual).
        notional (float): The principal amount used to calculate interest payments.
    """

    def __init__(self, fixed_rate: float, maturity: float, freq: int = 1, notional: float = 1.0):
        """
        Initializes the Interest Rate Swap with payment and rate parameters.
        """
        self.fixed_rate = fixed_rate
        self.T = maturity
        self.freq = freq
        self.notional = notional

    def discount_factor(self, r, t):
        """
        Calculates the continuous discount factor for a given time and rate.

        Formula: P(0, t) = exp(-r * t)
        """
        return np.exp(-r * t)

    def payment_times(self):
        """
        Generates the schedule of payment dates based on frequency and maturity.

        Returns:
            np.ndarray: Array of payment times (in years).
        """
        n = int(self.T * self.freq)
        return np.arange(1, n + 1) / self.freq

    def par_rate(self, r):
        """
        Computes the Par Swap Rate.

        The par rate is the fixed rate that makes the current market value
        of the swap equal to zero.

        Args:
            r (float): Current annualized risk-free interest rate (flat curve).

        Returns:
            float: The theoretical par rate.
        """
        times = self.payment_times()
        dfs = self.discount_factor(r, times)
        annuity = np.sum(dfs) / self.freq
        return (1.0 - self.discount_factor(r, self.T)) / annuity

    def value(self, r, payer=True):
        """
        Calculates the present value (PV) of the swap.

        The value is the difference between the PV of the floating leg
        and the PV of the fixed leg.

        Args:
            r (float): Current annualized interest rate.
            payer (bool): If True, represents the 'payer swap' (pays fixed, receives float).
                         If False, represents the 'receiver swap'.

        Returns:
            float: The net present value (NPV) multiplied by the notional.
        """
        times = self.payment_times()
        dfs = self.discount_factor(r, times)

        fixed_leg = self.fixed_rate * np.sum(dfs) / self.freq
        float_leg = 1.0 - self.discount_factor(r, self.T)

        pv = self.notional * (float_leg - fixed_leg)

        return pv if payer else -pv
