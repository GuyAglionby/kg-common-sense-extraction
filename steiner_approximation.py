import heapq
from networkx.utils import UnionFind


from itertools import chain


from networkx.utils import pairwise, not_implemented_for

from heapq import heappush, heappop
import networkx as nx
__all__ = ["metric_closure", "steiner_tree"]


class HeapEntry:
    def __init__(self, comparator, data):
        self.comparator = comparator
        self.data = data

    def __lt__(self, other):
        return self.comparator < other.comparator


@not_implemented_for("directed")
def metric_closure(G, weight="weight"):
    """Return the metric closure of a graph.

    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.

    """
    M = nx.Graph()

    Gnodes = set(G)

    # check for connected graph while processing first node
    all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
    u, (distance, path) = next(all_paths_iter)
    if Gnodes - set(distance):
        msg = "G is not a connected graph. metric_closure is not defined."
        raise nx.NetworkXError(msg)
    Gnodes.remove(u)
    for v in Gnodes:
        M.add_edge(u, v, distance=distance[v], path=path[v])

    # first node done -- now process the rest
    for u, (distance, path) in all_paths_iter:
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])

    return M


def kou_steiner_tree(G, terminal_nodes, weight):
    # H is the subgraph induced by terminal_nodes in the metric closure M of G.
    M = metric_closure(G, weight=weight)
    H = M.subgraph(terminal_nodes)
    # Use the 'distance' attribute of each edge provided by M.
    mst_edges = nx.minimum_spanning_edges(H, weight="distance", data=True)
    # Create an iterator over each edge in each shortest path; repeats are okay
    edges = chain.from_iterable(pairwise(d["path"]) for u, v, d in mst_edges)
    return edges


def mehlhorn_steiner_tree(G, S, weight):
    paths = nx.multi_source_dijkstra_path(G, S)

    d_1 = {}
    s = {}
    for v in G.nodes():
        s[v] = paths[v][0]
        d_1[(v, s[v])] = len(paths[v]) - 1

    new_edges = {}
    for e in G.edges(data=True):
        u, v, data = e
        weight_here = d_1[(u, s[u])] + data.get(weight, 1) + d_1[(v, s[v])]
        key = tuple(sorted([s[u], s[v]]))
        new_edges[key] = min(new_edges.get(key, float('inf')), weight_here)

    G_1_prime = nx.Graph()
    for (n1, n2), edge_weight in new_edges.items():
        G_1_prime.add_edge(n1, n2, weight=edge_weight)

    G_2 = nx.minimum_spanning_edges(G_1_prime, data=True)

    G_3 = nx.Graph()
    for u, v, d in G_2:
        path = next(nx.shortest_simple_paths(G, u, v))
        for n1, n2 in pairwise(path):
            G_3.add_edge(n1, n2)

    G_4 = nx.minimum_spanning_edges(G_3, data=False)
    return G_4


def wu_steiner_tree(G, terminal_nodes, weight):
    # Step 1
    source = {q: q for q in terminal_nodes}
    length = {q: 0 for q in terminal_nodes}
    pred = {}

    # Step 2
    priority_queue = []
    for q in terminal_nodes:
        for r in G.neighbors(q):
            # If q and r are in terminal_nodes, we only want to add one to the heap. Break ties arbitrarily.
            if r in terminal_nodes and r < q:
                continue
            d_q_r = G[q][r].get(weight, 1)
            heappush(priority_queue, HeapEntry(d_q_r, (r, d_q_r, q, None, None)))

    # Step 3
    subtrees = UnionFind(terminal_nodes)
    n_subtrees = len(terminal_nodes)

    generalised_mst_edges = []

    # Step 4
    while n_subtrees != 1:
        min_d = heappop(priority_queue)
        t, d, s, p1, p2 = min_d.data
        if t not in source:
            # case 1
            source[t] = s
            if t not in pred:
                pred[t] = s
            length[t] = d
            for r in G.neighbors(t):
                if r not in source:
                    d_t_r = G[t][r].get(weight, 1)
                    heap_dist = d + d_t_r
                    pred[r] = t
                    heappush(priority_queue, HeapEntry(heap_dist, (r, heap_dist, s, t, None)))
        elif subtrees[source[t]] == subtrees[s]:
            # case 2
            continue
        else:
            # case 3
            if t in terminal_nodes:
                # case 3.1
                subtrees.union(t, s)
                generalised_mst_edges.append((s, t, p1, p2))
                n_subtrees -= 1
            else:
                # case 3.2
                heapq.heappush(priority_queue, HeapEntry(d + length[t], (source[t], d + length[t], s, p1, t)))

    # The edges identified in the generalised minimum spanning tree are paths in the original graph.
    # Here, we find those paths (section 3).
    steiner_tree_edges = []
    for s, t, p1, p2 in generalised_mst_edges:
        # Follow the heads back
        for iterator in [p1, p2]:
            while iterator in pred:
                steiner_tree_edges.append((iterator, pred[iterator]))
                iterator = pred[iterator]

        # Deal with where the heads meet (figure 2 and section 3 paragraph 2)
        if p1 is None:
            p1 = s

        if t in terminal_nodes:
            if p2 is None:
                p2 = t
            steiner_tree_edges.append((p1, p2))
        else:
            steiner_tree_edges.append((p1, p2))
            while t in pred:
                steiner_tree_edges.append((t, pred[t]))
                t = pred[t]

    return steiner_tree_edges


ALGORITHMS = {
    "wu": wu_steiner_tree,
    "kou": kou_steiner_tree,
    "mehlhorn": mehlhorn_steiner_tree
}


@not_implemented_for("directed")
def steiner_tree(G, terminal_nodes, weight="weight", algorithm="kou"):
    """Return an approximation to the minimum Steiner tree of a graph.

    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes`
    is a tree within `G` that spans those nodes and has minimum size
    (sum of edge weights) among all such trees.

    The minimum Steiner tree can be approximated by computing the minimum
    spanning tree of the subgraph of the metric closure of *G* induced by the
    terminal nodes, where the metric closure of *G* is the complete graph in
    which each edge is weighted by the shortest path distance between the
    nodes in *G* .
    This algorithm produces a tree whose weight is within a (2 - (2 / t))
    factor of the weight of the optimal Steiner tree where *t* is number of
    terminal nodes.

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    weight : None or string, optional (default = 'weight')
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.

    algorithm : string, optional (default = 'kou')
        The algorithm to use to approximate the Steiner tree.
        Supported options: 'kou', 'wu'.
        Other inputs produce a ValueError.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    Notes
    -----
    For multigraphs, the edge between two nodes with minimum weight is the
    edge put into the Steiner tree.


    References
    ----------
    .. [1] Steiner_tree_problem on Wikipedia.
       https://en.wikipedia.org/wiki/Steiner_tree_problem
    .. [2] L Kou, G Markowsky, and L Berman. 1981. A fast algorithm for Steiner trees.
    .. [3] K Mehlhorn. 1988. A faster approximation algorithm for the Steiner problem in graphs.
    .. [4] Y F Wu, P Widmayer, and C K Wong. 1986. A faster approximation algorithm for the Steiner problem in graphs.


    """
    try:
        algo = ALGORITHMS[algorithm]
    except KeyError as e:
        msg = f"{algorithm} is not a valid choice for an algorithm."
        raise ValueError(msg) from e

    edges = algo(G, terminal_nodes, weight)
    # For multigraph we should add the minimal weight edge keys
    if G.is_multigraph():
        edges = (
            (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in edges
        )
    T = G.edge_subgraph(edges)
    return T
