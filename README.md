# Optimized Dumpster Siting Based on Demand

## Background

This project was inspired by a previous project completed by my team in the class 1.101, Introduction to Civil and Environmental Engineering Design. That project looked at creatign a prototype autonomous dumpster that would convievably replace garbage trucks and gousehold track carts in the city of Cambridge's garbage collection system. The original premise of this project was to look at where those autonomous dumpsters would be placed in the city to optimally meet community needs, and where garbage accumulation sites could be placed. Currently, we have yet to consider the dynamic element of the autonomous dumpsters; we are only considering static distributions of dumpsters of varying capacities and optimizing for the best static arrangement. Further work would include adding complexity to the optimization and adding reatures such as:
- Dumpster routing
- Garbage accumulation sites ("mother dumpsters")

## Optimization Problem Formulation

Currently we are looking at dumpster siting as a two-level optimization problem:
- Determining the minimum flow given a proposed dumpster arrangement. The "flow" we are considering is that of residents potentially needing to dispose of their garbage at a dumpster further than the closest one because the closest dumpster is already full.
- Determining the dumpster arrangement that minimizes the optimal flow and satisfies other conditions.

### Minimum Flow

Minimum flow on a graph is a well-studied problem that we have not innovated upon. In order to accomodate differences between supply and demand, we have attempted to implement an "approximate minimum flow" solver because existing implementations such as networkx require total supply to equal total demand.

Let's represent the city network as a directed graph $G = (N, E)$ in which nodes represent road crossings where a dumpster could be placed. 
- Each node $n$ has a dumpster capacity $m_n$ and a supply $s_n.$ A positive supply indicates that a node has excess dumpster capacity compared to the amount of garbage expected to be produced at that location, and thus wants to "supply" capacity to other nodes. Negative supply indicates insufficient dumpster capacity. 
- Each edge $e$ has an edge capacity $u_e$ and cost $c_e$. The flow on any edge cannot exceed $u_e.$ We want to minimize, for our flow, the sum of flow times cost for all edges in our network.

Let $\vec f$ represent our flow, so that $f_e$ is the amount of flow that passes along edge $e.$ In our case, this would be the volume of garbage capacity (in arbitrary units) at node $e_{start}$ that is used by residents whose closest dumpster is at node $e_{end}.$

Let $A$ be an edge-node matrix where
$$A_{ij} = \left\lbrace\begin{matrix} +1 & \text{if edge}\ j\ \text{starts at node} \ i \\  -1 & \text{if edge}\ j\ \text{ends at node} \ i \\ 0 & \text{otherwise}.\end{matrix}\right. $$

Let $\vec c$ be a vector where $c_e$ is the cost of edge $e.$ Let $\vec s$ be a vector where $s_n$ is the supply at node $n$. Let $t$ be the tolerance of our minimum cost flow: the maximum discrepancy between the required supply at a node and the supply provided by our flow. I'll overload the subtraction operator $-$ so that subtracting a number from a vector indicates subtracting the same constant value from each element of the vector.

Then our minimum-cost flow problem can be formulated as such:

Minimize
$$\sum_{e\in E} c_ef_e = \vec c\cdot\vec f$$
given
$$f_e \leq u_e,\quad \forall e\in E$$
and
$$\vec s - t \leq A\vec f\leq \vec s + t.$$

The current method for solving this problem is using SLSQP as implemented in `scipy`. The next step would be to implement a deterministic constrained least-squares implementation that would minimize the supply-flow discrepancy with certainty.

### Dumpster Siting

Given a given dumpster distribution $\vec x$ where $x_i$ is the number of dumpsters on node $i$, the capacity of each dumpster, and the total number of dumpsters, we can calculate the number of dumpsters on each node and, using demand values for those nodes, calculate the minimum flow cost to approximately satisfy the garbage demand.

We approximate node demand by summing the volumes of all the buildings on the roads leading to that node. Volumes are divided equally between the two endpoint nodes of a road. We calculate building volume by taking the area of the building footprint and multiplying by the number of stories of the building.

The second level of optimization is to find that $\vec x$ that minimizes our multi-part objective that consists of the following factors:
- Mismatches between supply and demand. If the total supply to a node (dumpsters located at the node plus flow to the node) is not equal to that node's demand, we punish the optimizer.
- Flow cost. We want to minimize the amount of "space borrowing" required to satisfy the network's needs.
- Overloading. We add a constant value to the cost for every dumpster located at a node exceeding that node's dumpster capacity.

## Examples

Graphs can be visualized as follows:
- Points represent nodes in the network. 
  - The size of the node indicates the amount of demand.
  - The number inside the node indicates the amount of dumpsters assigned to that node.
  - The color of the node indicates the amount of space-borrowing that node had to engage in; dark red means grater borrowing, while dark blue means the node donated a large amount of space.
- Edges represent roads in the network.
  - The thickness of the edge represents the amount of space-borrowing that took place over this road.


The following two images show a grid network before and after dumpster location optimization; initialy, all the dumpsters are placed on the first row.

![1](/img/grid1.png)
![1opt](/img/grid1-opt.png)
