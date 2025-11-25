from pricer.market.data import MarketData
from pricer.models.monte_carlo import MonteCarloModel
from pricer.products.vanilla import EuropeanCall

market = MarketData(spot=31.55, r=0.05)
option = EuropeanCall(K=22.75, T=3.5)

model = MonteCarloModel(sigma=0.50)
price = model.price(option, market)

print(price)
