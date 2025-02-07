from typing import List

import csv, asyncio
import sys, time
import argparse

import numpy as np 

import networkx as nx


from src.causal_discovery import LLMCausalOrderSearcher
from src.dataset.dataset import Dataset
from src.dataset.ground import *
from src.utils import utils
import matplotlib.pyplot as plt


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
        '--temperature',
        type=float,
        default=0.7
    )   
    parser.add_argument(
        '--triplets',
        type=bool,
        default=False
    )
    
    return parser.parse_args()

async def main(args):
    dataset = Dataset(args.dataset)
    true_graph = dataset.graph

    # searcher
    searcher = LLMCausalOrderSearcher(
        model=args.model,
        dataset=dataset,
        temperature=0.7,
        triplets=args.triplets,
    )

    # search
    t0 = time.time()
    causal_orders, bidirected_nodes = await searcher.search()
    t1 = time.time()

    # ground truth
    print('bidirected_nodes', bidirected_nodes)
    for node in bidirected_nodes:
        true_graph.remove_node(node)

    # evaluation
    results = {
        'method': 'lcos',
        'dataset': args.dataset,
        'model': args.model,
        'topo_error': [],
        'topo_norm_error': [],
        'order_error': [],
        'order_norm_error': [],
        'time': t1 - t0,
        'temperature': args.temperature,
        'ciclic': np.any([order.is_ciclic() for order in causal_orders]),
        'triplets': args.triplets,
    }

    graph_consistency = [order.get_score() for order in causal_orders]  
    causal_orders = [order.to_nx() for order in causal_orders]  
    topo_error, topo_norm_error, order_error, order_norm_error = utils.eval_causal_order(true_graph, causal_orders)
    results['topo_error'] = topo_error
    results['topo_norm_error'] = topo_norm_error
    results['order_error'] = order_error
    results['order_norm_error'] = order_norm_error

    # print(np.mean(topo_error), np.std(topo_error))
    # print('best', np.min(topo_norm_error))
    # print('topo_error', np.mean(topo_norm_error), np.std(topo_norm_error))
    # print('orders', len(order_error))
    # print(np.mean(order_norm_error), np.std(order_norm_error))
    with open(f'results/text_driven.csv', 'a') as f:
        csv.writer(f).writerow(results.values())
            

if __name__ == '__main__':

    args = input_parser(sys.argv[1:])
    asyncio.run(main(args))


