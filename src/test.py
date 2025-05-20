import numpy as np
import matplotlib.pyplot as plt

x = np.random.beta(0.5, 0.5, 1000)
y = np.random.beta(0.5, 0.5, 1000)
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter plot of two beta-distributed variables')
plt.show()