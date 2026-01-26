class PricingEngine:
    """
    A universal interface for option pricing across different mathematical models.

    The PricingEngine acts as a high-level wrapper that dispatches pricing requests
    to the underlying model based on its capabilities (analytical or numerical).
    It abstracts away the specific method calls, providing a consistent API
    for the rest of the application.

    Attributes:
        model (object): An instance of a pricing model (e.g., BlackScholesModel,
                        MonteCarloModel, or HestonModel).
    """

    def __init__(self, model):
        """
        Initializes the pricing engine with a specific model.

        Args:
            model (object): The mathematical model to be used for pricing.
        """
        self.model = model

    def price_european(self, option, kind=None):
        """
        Prices a European option by automatically selecting the correct model method.

        This method uses introspection (hasattr) to determine how to call the
        underlying model:
        1. If the model has specific 'price_call' and 'price_put' methods
           (Analytical models), it uses the 'kind' argument to route the call.
        2. If the model has a generic 'price' method (Numerical/Simulation models),
           it passes the option object directly.

        Args:
            option (Option): The option contract object containing parameters like K and T.
            kind (str, optional): The type of option ('call' or 'put').
                                  Required for analytical models.

        Returns:
            float: The theoretical fair value of the option.

        Raises:
            ValueError: If 'kind' is missing or invalid for analytical models.
            TypeError: If the attached model does not implement a recognized pricing method.
        """
        if hasattr(self.model, "price_call") and hasattr(self.model, "price_put"):
            if kind == "call":
                return self.model.price_call(option.K, option.T)
            elif kind == "put":
                return self.model.price_put(option.K, option.T)
            else:
                raise ValueError("kind must be 'call' or 'put'")

        if hasattr(self.model, "price"):
            return self.model.price(option)

        raise TypeError("Model does not support pricing method.")
