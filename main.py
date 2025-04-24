import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

import scipy.optimize
import networkx as nx
import osmnx as ox

from functools import reduce

import approxflow

def factors(n):
    return reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if not n % i))

EPSILON = 10e-2

class N_Graph():
    """
    Class representation of a neighborhood graph.
    Contains nodes and edges. An assignment indicates how many dumpsters are placed at each node.
    Calculates the cost of a particular assignment based on the cost of the
    minimum flow satisfying all demands and the standard deviation of the assignment
    (more uniform assignments are better).
    """

    def __init__(self, nodes):
        """
        Takes as input a list of nodes, which have demands, locations, and connected nodes.
        Creates edges based on the nodes.
        """

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
            
        raise IndexError(f"No edge named {name} in this graph.")
    
    def recalc_edges(self):
        self.edges = []

        for p1 in self.nodes:
            for p2 in p1.connected:
                new_edge = Edge(p1, p2)
                if new_edge not in self.edges:
                    self.edges.append(new_edge)

    def create_networkx_graph(self):
        """
        Creates a networkx graph representation of this object.
        The demands of the nodes in the networkx graph are the unfulfilled
        demand of the nodes in this object.

        Currently, all edges are bidirectional.
        """

        G = nx.DiGraph()

        for node in self.nodes:
            G.add_node(node.index, demand = node.demand - node.supply)

        for edge in self.edges:
            G.add_edge(
                edge.start.index, edge.end.index,
                capacity = int(1e12),
                weight = int(edge.length),
                )
            G.add_edge(
                edge.end.index, edge.start.index,
                capacity = int(1e12),
                weight = int(edge.length),
                )

        return G

    def determine_supply_flow(self, nx_graph):
        # ans = nx.min_cost_flow(nx_graph)

        # ans = nx.network_simplex(nx_graph)[1]

        tol = 5

        res, flow = approxflow.min_cost_flow(self, capacity_factor = 10, supply_tolerance = tol)

        while not res.success:
            tol += 10
            res, flow = approxflow.min_cost_flow(self, capacity_factor = 10, supply_tolerance = tol)

        print(res)

        return flow

    def set_supply(self, x, total_dumpsters, dumpster_volume):
        """
        Takes in an assignment as a distribution of dumpsters and the total amount of dumpsters.

        Each node with dumpsters on it gets enough supply as the number of dumpsters there, or
        as must as its demand, whichever is less.

        We then create a networkx graph where each node's demand is the remaining unfulfilled demand.

        We solve the minimum flow problem to determine how much supply is borrowed from other nodes.

        These properties are assigned to the nodes.
        """
        # each dumpster will give enough supply to the node it's on as that node's demand
        # if there is extra supply, it will be distributed among the nodes connected to that node

        self.reset_supply()

        self.assignment = [round(prop * total_dumpsters) for prop in x / sum(x)]

        if sum(self.assignment) != total_dumpsters:
            self.assignment[-1] += total_dumpsters - sum(self.assignment)

        # print(f"Total supply: {dumpster_volume * sum(self.assignment)}")
        # print(f"Total demand: {sum(node.demand for node in self.nodes)}")

        for node_index, num_dumpsters in enumerate(self.assignment):
            if num_dumpsters: # if there's a dumpster there, set the node's supply
                cur_node = self.nodes[node_index]
                cur_node.supply = dumpster_volume * num_dumpsters

        self.G = self.create_networkx_graph()
        
        self.flow = self.determine_supply_flow(self.G)

        for u in self.flow:
            for v in self.flow[u]:
                f = self.flow[u][v]

                u_ = self.get_by_ind(u)
                v_ = self.get_by_ind(v)

                u_.supply -= f
                v_.supply += f

                if v_.dist(u_) in v_.borrowed_supply:
                    v_.borrowed_supply[v_.dist(u_)] += f
                else:
                    v_.borrowed_supply[v_.dist(u_)] = f

    def reset_supply(self):
        self.assignment = np.array([0] * len(self.nodes))

        for node in self.nodes:
            node.supply = 0
            node.borrowed_supply = {}

    def cost(self):
        """
        The cost of the assignment is the sum of two components:

        1. The sum of borrowed supply multiplied by the distance over which the supply was borrowed
        2. The standard deviation of the assignment
        3. A constant factor of 100 added for each dumpster that a node has over its max
            dumpster capacity
        """

        ans = 0

        for i, node in enumerate(self.nodes):
            # mismatches between supply and demand as a proportion of demand
            ans += abs((node.supply - node.demand)) / (node.demand if node.demand else 1)

            # people having to travel some distance to throw away their garbage
            # scaled by demand
            ans += sum(k * v for k, v in node.borrowed_supply.items()) / (node.demand if node.demand else 1)

            if self.assignment[i] > node.max_dumpsters:
                ans += 100 * self.assignment[i] - node.max_dumpsters

        # factor for uniformity of distribution

        # ans += np.std(self.assignment) / len(self.assignment)

        ans = float(ans)

        return ans
    
    def try_x(self, x, total_dumpsters, dumpster_volume):
        """
        Try an assignment, return the cost of this assignment, and then automatically
        reset the N_Graph network.
        """

        self.set_supply(x, total_dumpsters, dumpster_volume)

        ans = self.cost()

        self.reset_supply()

        return ans

    def plot(self):
        """
        Represent the current graph and assignment in a MatPlotLib chart.

        Nodes are represented as circles; nodes with borrowed supply are red.

        The size of the node represents its demand.

        Red rectangles around nodes represent, by their size, the number of dumpsters placed at
        that node.
        """

        fig, ax = plt.subplots()
        cmap = mpl.colormaps["seismic"]

        for edge in self.edges:
            ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], "tab:blue", zorder = -1)
        
        for i, node in enumerate(self.nodes):
            size = 25 + 100 * node.demand/self.max_demand()
            color = sum(node.borrowed_supply.values())/(node.demand + 0.1) + 0.5
            ax.scatter(node.x, node.y, s = size, color = cmap(color), edgecolor = "k", zorder = 1)

            if self.assignment[i]:
                ax.scatter(node.x, node.y, s = 2 * self.assignment[i], marker = "s", edgecolor = "r", color = "none", alpha = 1, zorder = 2)


class Edge():
    """
    Represents a connection between nodes.

    Length can be assigned distinct from the Euclidean distance between the nodes.
    """

    def __init__(self, start, end, length = None, name = "", width = 10):
        self.start = start
        self.end = end
        self.name = name

        self.length = length if length is not None else self.start.dist(self.end)
        self.width = width

    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end) # or (self.start == other.end and self.end == other.start)
    
    def __hash__(self): return super().__hash__()

    def __repr__(self):
        return f"{self.name}: {self.start} --> {self.end}, {self.length}"

class Node():
    """
    Represents a location where dumpsters can be placed.

    Each node has:
    
    1. a demand representing the amount of waste projected to be produced at this location
    2. a supply representing the amount of waste capacity currently provided to this location
    3. borrowed_supply, a dictionary containing information about distances traveled to 
        gain extra waste capacity to meet the demand. Keys are distances traveled and values
        are the amount of supply borrowed from that distance.
    4. a set of connected nodes
    5. a maximum number of dumpsters that can be accomodated at that node
    """

    def __init__(self, parent, loc, index = None, demand = 0, connected = None, max_dumpsters = 10):
        self.parent = parent
        self.x = loc[0]
        self.y = loc[1]

        self.demand = demand
        self.connected = connected if connected else set()
        self.index = index if index else self.parent.max_ind()

        self.supply = 0
        self.borrowed_supply = {}

        self.max_dumpsters = max_dumpsters

    def dist(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __eq__(self, other):
        return self.dist(other) < EPSILON

    def __hash__(self): return super().__hash__()

    def __str__(self):
        return f"{self.index}"

    def __repr__(self):
        return f"{self.index}"
        # return f"{self.index}: ({self.x: 0.2f}, {self.y: 0.2f}) - d = {self.demand}, s = {self.supply}, b = {self.borrowed_supply}"
        # return f"{self.index}: ({self.x: 0.2f}, {self.y: 0.2f}) - d = {self.demand - self.supply}, b = {self.borrowed_supply}"


def grid(xnum, ynum, xspace, yspace, demand = lambda index: 0):
    """
    Creates an N-Graph grid with a defined number of nodes and spacing in the
    x and y directions. Assigns demand to the nodes as a function of node index.
    """

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
    """
    Uses osmnx to create an N-Graph from a given address and a radius around the address.

    Nodes represent road intersections.

    Assigns demand to each node based on the number and size of buildings on the roads
    connected to that node.
    """

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

    for a in G.nodes:
        for b in G.nodes:
            if a != b:
                route = ox.routing.shortest_path(G, a, b, weight="length")
                # print(f"{route=}")

    ngraph.nodes = nodes

    for indstart, indend, data in G.edges(data = True):
        pstart = ngraph.get_by_ind(indstart)
        pend = ngraph.get_by_ind(indend)

        # print(data)

        road_width = data.get("width", 10)
        if isinstance(road_width, list):
            road_width = road_width[0]
        
        try:
            road_width = float(road_width)
        except:
            road_width = 10

        ngraph.edges.append(
            Edge(
                pstart,
                pend,
                data["length"],
                name = data.get("name", "unnamed"),
                width = road_width
            )
        )

        # ngraph.edges.append(
        #     Edge(
        #         pend,
        #         pstart,
        #         data["length"],
        #         name = data.get("name", "unnamed"),
        #         width = road_width
        #     )
        # )

        pstart.connected.add(pend)
        pend.connected.add(pstart)

    for index, node in enumerate(ngraph.nodes):
        node.index = index

    buildings = ox.features.features_from_place("Cambridge, Massachusetts, USA", {"building": True})

    # print(buildings)
    
    buildings_proj = ox.projection.project_gdf(buildings)

    areas = buildings_proj.area

    for index, row in buildings.iterrows():
        volume = float(row["building:levels"]) * areas[index]

        if not math.isnan(volume):
            try:
                building_edge = ngraph.get_edge_by_name(row["addr:street"])

                building_weight = int(round(volume/20) * 10)

                building_edge.start.demand += building_weight
                building_edge.end.demand += building_weight

            except IndexError:
                pass

    return ngraph

if __name__ == "__main__":
    #### Finding the optimal dumpster assignment for a location in Cambridge

    g = NGraph_from_location("564 Massachusetts Ave, Cambridge, Massachusetts, USA", 500)

    total_demand = sum(n.demand for n in g.nodes)


    # decide on the total number of dumpsters
    # to satisfy minimum flow, it must be a factor of the total demand
    demand_divisors = sorted(factors(total_demand))
    print(demand_divisors)
    dumpster_size_ind = int(len(demand_divisors) / 2) - 1

    dumpster_volume = demand_divisors[dumpster_size_ind]
    total_dumpsters = total_demand / dumpster_volume

    print(f"{dumpster_volume =}, {total_dumpsters =}")

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
        total_dumpsters = total_dumpsters,
        dumpster_volume = dumpster_volume
    )

    g.plot()

    # optimize using SLSQP

    # result = scipy.optimize.minimize(
    #         g.try_x,
    #         x0,
    #         args = (total_dumpsters, dumpster_volume),
    #         method = "SLSQP",
    #         bounds = bounds,
    #         # constraints=cstr,
    #         options = {
    #                 # 'rhobeg': 1/total_dumpsters,
    #                 'eps': 0.5/total_dumpsters,
    #                 'maxiter': 500,
    #                 },
    #     )

    print(result)

    g.set_supply(result.x, total_dumpsters, dumpster_volume)

    print(g.flow)

    print(g.assignment)
    # print(g.nodes)

    # # nx.draw(g.G)
    g.plot()