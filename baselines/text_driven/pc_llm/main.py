from typing import List
import time, asyncio
import csv, json
import sys, time
import argparse

import numpy as np 

import networkx as nx

from baselines.text_driven.pc_llm.cit import LLMBasedConditionalIndependece 
from baselines.text_driven.pc_llm.pc_llm import pc
from src.dataset.dataset import Dataset
from src.dataset.ground import *
import matplotlib.pyplot as plt
from src.utils.utils import eval_causal_order
from baselines.utils import orient_cpdag

def input_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='cancer'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini'
    )
    parser.add_argument(
        '--show',
        type=bool,
        default=False
    )        
    parser.add_argument(
        '--save',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7
    )
    parser.add_argument(
        '--languages',
        type=List[str],
        # default=['english', 'french', 'chinese']
        default=['english']
    )
    
    return parser.parse_args()

async def main(args):
    
    dataset = Dataset(args.dataset)
    true_graph = dataset.graph
    data = np.array(dataset.var_description).reshape(-1, 1).T 
    mapping = {i: var for i, var in enumerate(dataset.var_description)}
    t0 = time.time()
    cache = {
        'temperature': args.temperature,
        'dataset': args.dataset,
        'model': args.model
        }
    # save cache 
    with open("baselines/text_driven/pc_llm/cache.json", "w") as outfile: 
        json.dump(cache, outfile)
    cg = await pc(data, alpha=0.05, indep_test='llm')
    t1 = time.time()
    cg.to_nx_graph()
    g = cg.nx_graph
    g = nx.relabel_nodes(g, mapping)
    estimated_graphs = orient_cpdag(g)
    
    if args.dataset == 'neighborhood':
        bidirected_nodes = ['exposure to high criminality levels', 'lack of services']
    elif args.dataset == 'msu':
        bidirected_nodes = ['time to thrombolysis']
    else:
        bidirected_nodes = []

    for g in estimated_graphs:
        for node in bidirected_nodes:
            g.remove_node(node)
    
    for node in bidirected_nodes:
       true_graph.remove_node(node)

    topo_error, topo_norm_error, order_error, order_norm_error = eval_causal_order(true_graph, estimated_graphs)

    results = {
        'method': 'pc_llm',
        'model': args.model,
        'dataset': args.dataset,
        'topo_error': topo_error,
        'topo_norm_error': topo_norm_error,
        'order_error': order_error,
        'order_norm_error': order_norm_error,
        'time': t1 - t0,
        'temperature': 0.7,
        'ciclic': False,
        'triplets': False,
    }

    print(results)
    with open(f'results/text_driven.csv', 'a') as f:
        csv.writer(f).writerow(results.values())

if  __name__ == '__main__':
    args = input_parser(sys.argv[1:])
    asyncio.run(main(args))