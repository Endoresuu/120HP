class MarketData:
    """
    Données de marché "globales" :
    - spot : prix du sous-jacent
    - r    : taux sans risque
    - q    : taux de dividende (0 par défaut)
    """

    def __init__(self, spot: float, r: float = 0.04, q: float = 0.0):
        self.spot = spot
        self.r = r
        self.q = q

        # Alias pour compat avec ton vieux code (S_0 utilisé parfois)
        self.S_0 = spot
