import dumpsters as dp
import scipy
import numpy as np

#### Finding the optimal dumpster assignment for a location in Cambridge

g = dp.grid(
    5, 7, 10, 34, lambda index: 34 * np.sin(index) + 34
)

# g = NGraph_from_location("564 Massachusetts Ave, Cambridge, Massachusetts, USA", 500)

total_demand = sum(n.demand for n in g.nodes)

# # decide on the total number of dumpsters
# # to satisfy minimum flow, it must be a factor of the total demand
# demand_divisors = sorted(factors(total_demand))
# print(demand_divisors)
# dumpster_size_ind = int(len(demand_divisors) / 2) - 1

# dumpster_volume = demand_divisors[dumpster_size_ind]
# total_dumpsters = total_demand / dumpster_volume

# print(f"{dumpster_volume =}, {total_dumpsters =}")

bounds = scipy.optimize.Bounds(
        [0] * len(g.nodes),
        [1] * len(g.nodes),
    )

cstr = scipy.optimize.LinearConstraint(
    [1] * len(g.nodes),
    0.9,
    1.1,
    keep_feasible = False,
)

# create a random initial value for the optimization

start_offset = (np.random.rand(len(g.nodes)) - 0.5) * 0.25 * (1/len(g.nodes))

x0 = np.ones(len(g.nodes)) / len(g.nodes) + start_offset
x0 = x0 / sum(x0)

g.set_supply(
    x = x0,
    total_dumpsters = 50,
    dumpster_volume = 20
)

g.plot()

# optimize using SLSQP

result = scipy.optimize.minimize(
        g.try_x,
        x0,
        args = (total_dumpsters, dumpster_volume),
        method = "SLSQP",
        bounds = bounds,
        # constraints=cstr,
        options = {
                # 'rhobeg': 1/total_dumpsters,
                'eps': 0.5/total_dumpsters,
                'maxiter': 500,
                },
    )

print(result)

g.set_supply(result.x, total_dumpsters, dumpster_volume)

print(g.flow)

print(g.assignment)
# print(g.nodes)

# # nx.draw(g.G)
g.plot()