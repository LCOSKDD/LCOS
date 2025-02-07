import time, csv

from src.dataset.dataset import Dataset

import networkx as nx

import ges
import lingam
from baselines.data_driven.notears.linear import notears_linear
from baselines.data_driven.notears.nonlinear import NotearsMLP, notears_nonlinear
from src.utils.utils import eval_causal_order
from baselines.utils import orient_cpdag
from causallearn.search.ConstraintBased.PC import pc
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt 

class DataDrivenCausalDiscovery:

    def __init__(
        self,
        name: str = 'cancer',
        method: str = 'pc',
        linear: bool = False,
        normal: bool = False,    
    ) -> None: 
        
        self.name = name 
        self.method = method
        self.linear = linear
        self.normal = normal

        self.dataset = Dataset(name, linear=linear, normal=normal)
        
        match method:
            case 'pc':
                self.method = self._pc
            case 'lingam':
                self.method = self._lingam
            case 'notears':
                self.method = self._notears
            case 'ges':
                self.method = self._ges
            case _:
                raise NotImplementedError(f'{method} is not implemented')
            
        self.mapping = {i: v for i, v in enumerate(self.dataset.var_description)}
            
    def run(self):
        gs = self.method()
        transitive_gs = []
        for g in gs:
            g = nx.transitive_closure(g)
            transitive_gs.append(g)
        return transitive_gs
        
    def _pc(self):
        indep_test = 'fisherz'
        if not self.linear:
            indep_test = 'kci'
        g = pc(self.dataset.data.values, indep_test=indep_test)
        g.to_nx_graph()
        g = g.nx_graph
        g = nx.relabel_nodes(g, self.mapping)
        g = orient_cpdag(g)
        return g
    
    def _lingam(self):
        X = self.dataset.data.values
        model = lingam.DirectLiNGAM()
        model.fit(X)
        g = nx.from_numpy_array(model.adjacency_matrix_.T, create_using=nx.DiGraph())
        g = nx.relabel_nodes(g, self.mapping)
        return [g]

    def _notears(self):
        if self.linear:
            adj_g = notears_linear(self.dataset.data.values, 0.01, 'l2')
        else:
            print('here')
            n = self.dataset.data.shape[1]
            model = NotearsMLP(dims=[n, n*2, 1])
            scaler = MinMaxScaler()
            data = scaler.fit_transform(self.dataset.data.values.astype(float))
            adj_g = notears_nonlinear(model, data)

        g = nx.from_numpy_array(adj_g, create_using=nx.DiGraph())
        g = nx.relabel_nodes(g, self.mapping)
        return [g]
    
    def _ges(self):
        X = self.dataset.data.values
        estimate, score = ges.fit_bic(X)
        g = nx.from_numpy_array(estimate, create_using=nx.DiGraph())
        g = nx.relabel_nodes(g, self.mapping)
        g = orient_cpdag(g)
        return g
    

if __name__ == '__main__':

    import argparse
    from src.utils import utils

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='cancer'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pc'
    )
    parser.add_argument(
        '--linear',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--normal',
        type=bool,
        default=False
    )
    args = parser.parse_args()

    model = DataDrivenCausalDiscovery(
            name=args.dataset,
            method=args.method,
            linear=args.linear,
            normal=args.normal
        )
    
    t0 = time.time()
    estimated_graphs = model.run()
    t1 = time.time()

    if args.dataset == 'neighborhood':
        bidirected_nodes = ['exposure to high criminality levels', 'lack of services']
    elif args.dataset == 'msu':
        bidirected_nodes = ['time to thrombolysis']
    else:
        bidirected_nodes = []

    for g in estimated_graphs:
        for node in bidirected_nodes:
            g.remove_node(node)
    
    true_graph = model.dataset.graph
    for node in bidirected_nodes:
       true_graph.remove_node(node)

    topo_error, topo_norm_error, order_error, order_norm_error = eval_causal_order(true_graph, estimated_graphs)

    results = {
        'method': args.method,
        'dataset': args.dataset,
        'linear': args.linear,
        'normal': args.normal,
        'topo_error': topo_error,
        'topo_norm_error': topo_norm_error,
        'order_error': order_error,
        'order_norm_error': order_norm_error,
        'time': t1 - t0
    }

    print(results)
    with open(f'results/data_driven.csv', 'a') as f:
        csv.writer(f).writerow(results.values())

    
