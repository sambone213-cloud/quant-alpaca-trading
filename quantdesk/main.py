import matplotlib.pyplot as plt
from market import PredictionMarketABM
from filters import PredictionMarketParticleFilter

# 1. Simulate the noisy market
market = PredictionMarketABM(true_prob=0.65, n_informed=10, n_noise=50)
prices = market.run(500)

# 2. Filter the noise
brain = PredictionMarketParticleFilter(prior_prob=0.50)
filtered = []
for p in prices:
    brain.update(p)
    filtered.append(brain.estimate())

# 3. Visualize
plt.plot(prices, label='Market Price (Noisy)', alpha=0.5)
plt.plot(filtered, label='Filter Estimate (Smart)', color='red')
plt.axhline(0.65, color='green', linestyle='--', label='Truth')
plt.legend()
plt.show()