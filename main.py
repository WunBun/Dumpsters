import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from gekko import GEKKO

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
    
    def recalc_edges(self):
        self.edges = []

        for p1 in self.nodes:
            for p2 in p1.connected:
                new_edge = Edge(p1, p2)
                if new_edge not in self.edges:
                    self.edges.append(new_edge)

    def set_supply(self, x, dumpster_volume = 20):
        # each dumpster will give enough supply to the node it's on as that node's demand
        # if there is extra supply, it will be distributed among the nodes connected to that node
        # x = [v.value for v in x]


        self.reset_supply()
        self.assignment = x

        for i, num_dumpsters in enumerate(x):
            if np.floor(num_dumpsters): # if there's a dumpster there
                remaining_supply = dumpster_volume * np.floor(num_dumpsters)
                cur_node = self.get_by_ind(i)

                cur_node.supply += min(remaining_supply, cur_node.demand)
                remaining_supply -= min(remaining_supply, cur_node.demand)

                if remaining_supply:
                    quot, remain = divmod(remaining_supply, len(cur_node.connected))
                    for i, con_node in enumerate(cur_node.connected):
                        if i == 0:
                            con_node.supply += quot + remain
                            con_node.borrowed_supply[cur_node] = quot + remain
                            remaining_supply -= (quot + remain)
                        else:
                            con_node.supply += quot
                            con_node.borrowed_supply[cur_node] = quot
                            remaining_supply -= quot

    def reset_supply(self):
        for node in self.nodes:
            node.supply = 0
            node.borrowed_supply = {}

    def cost(self):
        ans = 0

        ans += sum(abs(node.demand - node.supply) for node in self.nodes)

        return float(ans)
    
    def try_x(self, x):
        self.set_supply(x)

        ans = self.cost()

        self.reset_supply()

        return ans

    def conn_mat(self):
        pass

    def plot(self):
        fig, ax = plt.subplots()
        cmap = mpl.colormaps["viridis"]

        for edge in self.edges:
            ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], "tab:blue")
        
        for i, node in enumerate(self.nodes):
            size = 50 + 100 * node.demand/self.max_demand()
            color = node.supply/node.demand / 2
            ax.scatter(node.x, node.y, s = size, color = cmap(color), alpha = 0.5)

            if self.assignment[i]:
                ax.scatter(node.x, node.y, s = 200, marker = "s", edgecolor = "r", color = "none", alpha = 1)


class Edge():
    # edge from node to node
    # yes it's redundant
    # i'll take it out later if not necessary

    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.length = self.start.dist(self.end)

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
        self.borrowed_supply = {}

    def dist(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __eq__(self, other):
        return self.dist(other) < EPSILON

    def __hash__(self): return super().__hash__()

    def __repr__(self):
        return f"{self.index}: ({self.x: 0.2f}, {self.y: 0.2f}) - d = {self.demand}, s = {self.supply}"


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

g = grid(3, 5, 10, 10, lambda i: 40 * (i % 2) + 5)

m = GEKKO(remote = False)

xs = m.Array(m.Var, len(g.nodes), value = 2, lb = 0, ub = 10, integer = True)
print(g.try_x(xs))
m.Minimize(g.try_x)

m.options.SOLVER = 3
m.solve(disp=True, debug=True)

print(xs)
print(g.try_x(xs))

# g.set_supply([1, 0, 0, 0, 1, 0, 0, 0, 1])