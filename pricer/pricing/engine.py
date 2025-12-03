class PricingEngine:
    """
    Moteur de pricing générique.
    Connecte un modèle (Black-Scholes, MonteCarlo, Heston)
    avec un produit (Option européenne).
    """

    def __init__(self, model):
        self.model = model

    def price_european(self, option, kind=None):
        """
        Pricing européen standard.
        kind = "call" ou "put"
        """

        # Cas des modèles Black-Scholes (méthodes call + put)
        if hasattr(self.model, "price_call") and hasattr(self.model, "price_put"):
            if kind == "call":
                return self.model.price_call(option.K, option.T)
            elif kind == "put":
                return self.model.price_put(option.K, option.T)
            else:
                raise ValueError("kind must be 'call' or 'put'")

        # Cas des modèles Monte Carlo / Heston / etc.
        if hasattr(self.model, "price"):
            return self.model.price(option)

        raise TypeError("Model does not support pricing method.")
