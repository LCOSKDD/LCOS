from src.dataset.dataset import Dataset
from src.llm.uncertain_expert import UncertainExpert
from src.utils import utils
from baselines.text_driven.llm_order.triplet_discovery import TripletDiscovery

import sys, time, asyncio
import argparse, csv
from tqdm import tqdm
import numpy as np 
import networkx as nx


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
    
    return parser.parse_args()

async def main(args):
    searcher = TripletDiscovery(args.model, args.dataset, args.temperature)
    t0 = time.time()
    G = await searcher.search()
    estimated_graphs = [G]
    t1 = time.time()
    true_graph = Dataset(args.dataset).graph  

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

    topo_error, topo_norm_error, order_error, order_norm_error = utils.eval_causal_order(true_graph, [G])
    # plot 
    # nx.draw(G, with_labels=True)   
    # import matplotlib.pyplot as plt
    # plt.show()
    
    results = {
        'method': 'triplet_llm',
        'model': args.model,
        'dataset': args.dataset,
        'topo_error': topo_error,
        'topo_norm_error': topo_norm_error,
        'order_error': order_error,
        'order_norm_error': order_norm_error,
        'time': t1 - t0,
        'temperature': 0.7,
        'ciclic': not nx.is_directed_acyclic_graph(G),
        'triplets': True,
    }

    print(results)
    with open(f'results/text_driven.csv', 'a') as f:
        csv.writer(f).writerow(results.values())


if __name__ == '__main__':
    args = input_parser(sys.argv[1:])
    asyncio.run(main(args))

