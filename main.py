import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

import scipy.optimize
import networkx as nx
import osmnx as ox

import random

EPSILON = 10e-2

class N_Graph():
    # Neighborhood graph
    # Holds nodes and edges

    def __init__(self, nodes):
        # takes as input an iterable of nodes

        self._nodes = list(nodes)
        self.edges = []
        self.assignment = np.array([0] * len(nodes))

        for p1 in self.nodes:
            for p2 in p1.connected:
                new_edge = Edge(p1, p2)
                if new_edge not in self.edges:
                    self.edges.append(new_edge)

    @property
    def nodes(self):
        return self._nodes
    
    @nodes.setter
    def nodes(self, new_nodes):
        self._nodes = new_nodes
        self.assignment = np.array([0] * len(self.nodes))

        self.edges = []
        for p1 in self.nodes:
            for p2 in p1.connected:
                new_edge = Edge(p1, p2)
                if new_edge not in self.edges:
                    self.edges.append(new_edge)

    def extend(self, iter):
        for node in iter:
            self.nodes.add(node)

        self.assignment = np.array([0] * len(self.nodes))

        for p1 in self.nodes:
            for p2 in p1.connected:
                new_edge = Edge(p1, p2)
                if new_edge not in self.edges:
                    self.edges.append(new_edge)

    def max_ind(self):
        if not self.nodes:
            return 0
        
        return max(node.index for node in self.nodes)
    
    def max_demand(self):
        if not self.nodes:
            return 0
        
        return max(node.demand for node in self.nodes)
    
    def get_by_ind(self, ind):
        for node in self.nodes:
            if node.index == ind:
                return node
            
        raise IndexError(f"No node with index {ind} in this graph")
    
    def get_edge_by_name(self, name):
        for edge in self.edges:
            if edge.name == name:
                return edge
            
        raise IndexError
    
    def recalc_edges(self):
        self.edges = []

        for p1 in self.nodes:
            for p2 in p1.connected:
                new_edge = Edge(p1, p2)
                if new_edge not in self.edges:
                    self.edges.append(new_edge)

    def create_networkx_graph(self):
        G = nx.DiGraph()

        for node in self.nodes:
            G.add_node(node, demand = node.demand - node.supply - node.extra_supply)

        for edge in self.edges:
            G.add_edge(edge.start, edge.end, weight = edge.length)
            G.add_edge(edge.end, edge.start, weight = edge.length)

        return G

    def determine_supply_flow(self, nx_graph):
        ans = nx.min_cost_flow(nx_graph)

        return ans

    def set_supply(self, x, total_dumpsters, dumpster_volume):
        # each dumpster will give enough supply to the node it's on as that node's demand
        # if there is extra supply, it will be distributed among the nodes connected to that node

        self.reset_supply()

        self.assignment = [round(prop * total_dumpsters) for prop in x / sum(x)]

        if sum(self.assignment) != total_dumpsters:
            self.assignment[-1] += total_dumpsters - sum(self.assignment)

        # print(self.assignment)
        # print(sum(self.assignment))
        # print(dumpster_volume)

        for node_index, num_dumpsters in enumerate(self.assignment):
            if num_dumpsters: # if there's a dumpster there
                remaining_supply = dumpster_volume * num_dumpsters
                cur_node = self.nodes[node_index]

                cur_node.supply += min(remaining_supply, cur_node.demand)
                remaining_supply -= min(remaining_supply, cur_node.demand)

                cur_node.extra_supply = remaining_supply

        self.G = self.create_networkx_graph()
        
        self.flow = self.determine_supply_flow(self.G)

        for u in self.flow:
            for v in self.flow[u]:
                f = self.flow[u][v]

                u.extra_supply -= f
                v.supply += f

                if v.dist(u) in v.borrowed_supply:
                    v.borrowed_supply[v.dist(u)] += f
                else:
                    v.borrowed_supply[v.dist(u)] = f

    def reset_supply(self):
        self.assignment = np.array([0] * len(self.nodes))

        for node in self.nodes:
            node.supply = 0
            node.extra_supply = 0
            node.borrowed_supply = {}

    def cost(self):
        ans = 0

        for node in self.nodes:
            # proportion of demand that is not met
            ans += (node.demand - node.supply) / node.demand

            # excess supply as a proportion of demand
            ans += node.extra_supply / node.demand

            # people having to travel some distance to throw away their garbage
            # scaled by demand

            ### fix this
            ans += sum(k * v for k, v in node.borrowed_supply.items()) / node.demand

        ans = float(ans)

        return ans
    
    def try_x(self, x, total_dumpsters, dumpster_volume):
        self.set_supply(x, total_dumpsters, dumpster_volume)

        ans = self.cost()

        self.reset_supply()

        return ans

    def plot(self):
        fig, ax = plt.subplots()
        cmap = mpl.colormaps["seismic"]

        for edge in self.edges:
            ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], "tab:blue", zorder = -1)
        
        for i, node in enumerate(self.nodes):
            size = 50 + 200 * node.demand/self.max_demand()
            color = sum(node.borrowed_supply.values())/node.demand + 0.5
            ax.scatter(node.x, node.y, s = size, color = cmap(color), edgecolor = "k", zorder = 1)

            if self.assignment[i]:
                ax.scatter(node.x, node.y, s = 100 * self.assignment[i], marker = "s", edgecolor = "r", color = "none", alpha = 1, zorder = 2)


class Edge():
    # edge from node to node
    # yes it's redundant
    # i'll take it out later if not necessary

    def __init__(self, start, end, length = None, name = ""):
        self.start = start
        self.end = end
        self.name = name

        self.length = length if length is not None else self.start.dist(self.end)

    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end) or (self.start == other.end and self.end == other.start)
    
    def __hash__(self): return super().__hash__()

    def __repr__(self):
        return f"{self.start} --> {self.end}"

class Node():
    # a node in the dumpster graph represents a crossroads?
    # a location where a dumpster can be placed

    def __init__(self, parent, loc, index = None, demand = 0, connected = None):
        self.parent = parent
        self.x = loc[0]
        self.y = loc[1]

        self.demand = demand
        self.connected = connected if connected else set()
        self.index = index if index else self.parent.max_ind()

        self.supply = 0
        self.extra_supply = 0
        self.borrowed_supply = {}

    def dist(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __eq__(self, other):
        return self.dist(other) < EPSILON

    def __hash__(self): return super().__hash__()

    def __str__(self):
        return f"{self.index}"

    def __repr__(self):
        # return f"{self.index}"
        # return f"{self.index}: ({self.x: 0.2f}, {self.y: 0.2f}) - d = {self.demand}, s = {self.supply}, ex = {self.extra_supply}, b = {self.borrowed_supply}"
        return f"{self.index}: ({self.x: 0.2f}, {self.y: 0.2f}) - d = {self.demand - self.supply - self.extra_supply}, b = {self.borrowed_supply}"


def grid(xnum, ynum, xspace, yspace, demand = lambda index: 0):
    nodes = []
    graph = N_Graph([])

    for r in range(ynum):
        for c in range(xnum):
            nodes.append(Node(graph, (xspace * c, -yspace * r), xnum * r + c, demand(xnum * r + c)))

    graph.nodes = nodes

    for i, node in enumerate(nodes):
        if i % xnum != xnum - 1:
            node.connected.add(graph.get_by_ind(i + 1))
            graph.get_by_ind(i + 1).connected.add(node)

        if i + xnum < xnum * ynum:
            node.connected.add(graph.get_by_ind(i + xnum))
            graph.get_by_ind(i + xnum).connected.add(node)

    graph.recalc_edges()

    return graph

def NGraph_from_location(loc_str, dist):
    G = ox.graph.graph_from_address(
    loc_str,
    dist,
    network_type="drive",
    simplify = True,
    )
    
    fig, ax = ox.plot.plot_graph(G)

    ngraph = N_Graph([])

    nodes = [
        Node(
            ngraph,
            (data["x"], data["y"]),
            n_ind,
            0,
            None
        )
        for n_ind, data in G.nodes(data = True)
    ]

    ngraph.nodes = nodes

    for indstart, indend, data in G.edges(data = True):
        pstart = ngraph.get_by_ind(indstart)
        pend = ngraph.get_by_ind(indend)

        ngraph.edges.append(
            Edge(
                pstart,
                pend,
                data["length"],
                data.get("name", "unnamed")
            )
        )

        pstart.connected.add(pend)
        pend.connected.add(pstart)

    # print(nodes)
    # print(ngraph.edges)

    buildings = ox.features.features_from_place(loc_str, {"building": True})
    
    buildings_proj = ox.projection.project_gdf(buildings)

    areas = buildings_proj.area

    for index, row in buildings.iterrows():
        volume = float(row["building:levels"]) * areas[index]

        building_edge = ngraph.get_edge_by_name(row["addr:street"])

        building_weight = round(volume/20) * 10

        building_edge.start.demand += building_weight
        building_edge.end.demand += building_weight

    return ngraph

g = NGraph_from_location("450 Memorial Drive, Cambridge, Massachusetts, USA", 500)


# look at background of minimum flow!! Networkx documentation

# g = grid(4, 4, 10, 5, lambda i: 10 + 30 * (i % 3))

total_demand = sum(n.demand for n in g.nodes)
print(total_demand)

total_dumpsters = total_demand / 10
dumpster_volume = total_demand / total_dumpsters

bounds = scipy.optimize.Bounds(
        [0] * len(g.nodes),
        [1] * len(g.nodes),
    )

cstr = scipy.optimize.LinearConstraint(
    [1] * len(g.nodes),
    1,
    1,
    keep_feasible = False,
)

# x0 = [1/len(g.nodes)] * len(g.nodes)
x0 = [1] + [0] * (len(g.nodes) - 1)

result = scipy.optimize.minimize(
        g.try_x,
        x0,
        args = (total_dumpsters, dumpster_volume),
        method = "SLSQP",
        bounds = bounds,
        constraints=cstr,
        options = {
                   'rhobeg': 1/total_dumpsters,
                   'eps': 1/total_dumpsters,
                   'maxiter': 1000,
                   },
    )

print(result)

g.set_supply(result.x, total_dumpsters, dumpster_volume)

print(g.assignment)
# print(g.nodes)

# nx.draw(g.G)
g.plot()