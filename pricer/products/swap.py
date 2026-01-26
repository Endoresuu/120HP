import numpy as np

class InterestRateSwap:
    """
    Plain vanilla fixed-for-floating interest rate swap.
    Assumptions:
    - flat interest rate r
    - annual payments
    - floating rate = r
    """

    def __init__(self, fixed_rate: float, maturity: float, freq: int = 1, notional: float = 1.0):
        self.fixed_rate = fixed_rate
        self.T = maturity
        self.freq = freq
        self.notional = notional

    def discount_factor(self, r, t):
        return np.exp(-r * t)

    def payment_times(self):
        n = int(self.T * self.freq)
        return np.arange(1, n + 1) / self.freq

    def par_rate(self, r):
        times = self.payment_times()
        dfs = self.discount_factor(r, times)
        annuity = np.sum(dfs) / self.freq
        return (1.0 - self.discount_factor(r, self.T)) / annuity

    def value(self, r, payer=True):
        """
        payer=True : pay fixed, receive float
        payer=False: receive fixed, pay float
        """
        times = self.payment_times()
        dfs = self.discount_factor(r, times)

        fixed_leg = self.fixed_rate * np.sum(dfs) / self.freq
        float_leg = 1.0 - self.discount_factor(r, self.T)

        pv = self.notional * (float_leg - fixed_leg)

        return pv if payer else -pv
