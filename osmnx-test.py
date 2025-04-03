import networkx as nx
import osmnx as ox

G = ox.graph.graph_from_address(
    "450 Memorial Drive, Cambridge, Massachusetts, USA",
    500,
    network_type="drive",
    simplify = True,
    )
fig, ax = ox.plot.plot_graph(G)
print(G)