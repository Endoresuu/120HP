class MarketOption:
    """
    Option contenant les paramètres nécessaires pour le calcul de volatilité implicite.
    """
    def __init__(self, S0, K, T, r, price_mkt):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.price_mkt = price_mkt
