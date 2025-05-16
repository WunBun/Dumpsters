import dumpsters as dp
import scipy
import numpy as np

g = dp.grid(
    5, 7, 70, 35, lambda index: 100 * np.sin(index) + 34
)

n_dumpsters = 100
d_capacity = 10

g.set_supply(np.array([1] * 5 + [0]*30), n_dumpsters, d_capacity)

f = g.plot()

res = dp.optimal_distribution(g, n_dumpsters, d_capacity)

print(res)

g.set_supply(res.x, n_dumpsters, d_capacity)

g.plot()