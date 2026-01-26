from abc import ABC, abstractmethod

class Option(ABC):
    """
    Abstract Base Class for all financial option contracts.

    This class defines the shared structure for any option type, ensuring that
    all subclasses implement a payoff calculation. It cannot be instantiated
    directly.

    Attributes:
        K (float): The strike price (exercise price) of the option.
        T (float): The time to maturity in years.
    """

    def __init__(self, strike: float, maturity: float):
        """
        Initializes the option with its core contractual parameters.

        Args:
            strike (float): The strike price of the option.
            maturity (float): The time until expiration, expressed in years.
        """
        self.K = strike
        self.T = maturity

    @abstractmethod
    def payoff(self, ST: float) :
        """
        Abstract method to calculate the option's value at expiration.

        This method must be overridden by any concrete subclass (e.g., Call or Put).
        It defines the cash flow the option holder receives based on the terminal
        price of the underlying asset.

        Args:
            ST (float | np.ndarray): The price of the underlying asset at maturity.
                                     Can be a single value or a NumPy array of values.

        Returns:
            float | np.ndarray: The non-negative payoff of the option.
        """
        pass
