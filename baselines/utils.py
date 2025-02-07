import networkx as nx 

def orient_cpdag(G: nx.DiGraph):
        queue = [G]
        undirected_edges = []
        for node_i in G.nodes:
            for node_j in G.nodes:
                if node_i != node_j:
                    if G.has_edge(node_i, node_j) and G.has_edge(node_j, node_i):
                        uedge = [node_i, node_j]
                        uedge.sort()
                        undirected_edges.append(uedge) if uedge not in undirected_edges else undirected_edges
        while undirected_edges:
            uedge = undirected_edges.pop(0)
            to_orient = []
            for G in queue:
                G_copy = G.copy()
                u, v = uedge[0], uedge[1]
                G.remove_edge(u, v)
                collider = False
                for node in nx.predecessor(G, v):
                    if is_unshilded_collider(G, u, node, v):
                        collider = True
                        break
                to_orient.append(G) if not collider else to_orient
                G_copy = G.copy()
                G_copy.remove_edge(v, u)
                collider = False
                for node in nx.predecessor(G, u):
                    if is_unshilded_collider(G, v, node, u):
                        collider = True
                        break
                to_orient.append(G_copy) if not collider else to_orient
            queue = to_orient
        oriented_graphs = []
        for G in queue:
            oriented_graphs.append(G) if nx.is_directed_acyclic_graph(G) else oriented_graphs
        return queue

def is_unshilded_collider(G: nx.DiGraph, x, y, z):
    return (G.has_edge(x, z) and G.has_edge(y, z) and not G.has_edge(x, y))
    