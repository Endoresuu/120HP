class MarketData:
    """
    Données de marché globales :
    - spot : prix du sous-jacent
    - r    : taux sans risque
    - q    : taux de dividende (0 par défaut)

    Et on laisse la place aux paramètres Heston.
    """

    def __init__(self, spot: float | None = None, r: float = 0.04,
                 q: float = 0.0, T: float | None = None):
        self.spot = spot
        self.S_0 = spot          # alias pour vieux code
        self.r = r
        self.q = q
        self.T = T

        # Initialisation Heston par défaut
        self.v_0 = None
        self.kappa = None
        self.theta = None
        self.sigma_v = None
        self.rho = None
        self.n_steps = None
        self.n_paths = None
