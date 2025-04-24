import scipy
import scipy.optimize
import numpy as np

def equality_matrix(nodes, edges):
    ans = []

    for node in nodes:
        row = []
        for edge in edges:
            if edge.start == node:
                row.append(1)
            elif edge.end == node:
                row.append(-1)
            else:
                row.append(0)
        ans.append(row)

    return np.array(ans)

def flow_cost(flow_vec, c):
    # c, A_supply, supplies = args

    # cost of flow along each edge

    flow_cost = c @ flow_vec.transpose()

    # cost of mismatches between supply and demand

    # print(supplies)
    # print(flow_vec)
    # print(np.ones((1, supplies.shape[1])))
    # print((A_supply @ flow_vec.transpose() - supplies.transpose()))

    # mismatch_cost = np.ones((1, supplies.shape[1])) @ abs((A_supply @ flow_vec.transpose() - supplies.transpose()))

    return flow_cost.item(0)


def min_cost_flow(NGraph, capacity_factor = 10, supply_tolerance = 2):
    """
    Determine an approximate minimum cost flow that satisfies
    some demand to within a set tolerance
    """

    nodes = NGraph.nodes
    edges = NGraph.edges

    u = np.array([capacity_factor * edge.width for edge in edges])
    c = np.array([edge.length for edge in edges], ndmin = 2)

    supplies = np.array([node.supply - node.demand for node in nodes])

    A_supply = equality_matrix(nodes, edges)

    bounds = scipy.optimize.Bounds(
        lb = np.zeros(len(edges)),
        ub = u,
        keep_feasible = False
    )

    constraints = scipy.optimize.LinearConstraint(
        A = A_supply,
        lb = supplies - supply_tolerance,
        ub = supplies + supply_tolerance
    )

    result = scipy.optimize.minimize(
        fun = flow_cost,
        x0 = np.ones((len(edges))),
        args = (c),
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints
    )

    flow = {}

    for i, edge in enumerate(edges):
        if result.x[i]:
            if edge.start not in flow:
                flow[edge.start] = {}

            flow[edge.start][edge.end] = result.x[i]

    return result, flow

if __name__ == "__main__":
    from main import grid

    g = grid(
        3, 3, 30, 50, lambda i: 10 * np.sin(i)
    )
    total_demand = sum(n.demand for n in g.nodes)
    for n in g.nodes:
        n.supply = total_demand / len(g.nodes)

    g.nodes[-1].supply += 1

    res, flow = min_cost_flow(g)

    print(res)
    print(flow)