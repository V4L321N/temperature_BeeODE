from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mu, sigma = 1, 0.33
NOISE = np.random.normal(mu, sigma, 10000)#000)

shape, scale = 4, 0.3
NOISE2 = np.random.gamma(shape, scale, 10000)

# count, bins, ignored = plt.hist(NOISE, 50)
count, bins, ignored = plt.hist(NOISE2, 100)


plt.plot(bins)
plt.xlim(-3,3)
plt.show()
