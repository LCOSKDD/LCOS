import networkx as nx
import numpy as np
from src.utils.metrics import TopologicalOrderError, CausalOrder


def compute_subset(A, res, subset, index):
    res.append(subset[:])
    for i in range(index, len(A)):
        subset.append(A[i])
        compute_subset(A, res, subset, i + 1)
        subset.pop()


def match_var(var, var_names, var_description):
    for i, name in enumerate(var_names):
        if var.lower() == name.lower():
            return i
    for i, desc in enumerate(var_description):
        if var.lower() in desc.lower():
            return i


def subsets(A):
    subset = []
    res = []
    index = 0
    compute_subset(A, res, subset, index)
    return res


def eval_causal_order(true_graph: nx.DiGraph, estimated_graphs: nx.DiGraph):

    topo_metric = TopologicalOrderError()
    order_metric = CausalOrder()

    topo_errors, topo_norm_errors, order_errors, order_norm_errors = [], [], [], []
    for estimated_graph in estimated_graphs:
        transitive_closure = nx.transitive_closure(true_graph)
        topo_error, topo_norm_error = topo_metric(true_graph, estimated_graph)
        order_error, order_norm_error = order_metric(transitive_closure, estimated_graph)
        topo_errors.append(topo_error)
        topo_norm_errors.append(topo_norm_error)
        order_errors.append(order_error)
        order_norm_errors.append(order_norm_error)

    return topo_errors, topo_norm_errors, order_errors, order_norm_errors