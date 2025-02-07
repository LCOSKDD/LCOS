from src.dataset.dataset import Dataset
from src.llm.uncertain_expert import UncertainExpert
from src.utils import utils

import time, asyncio
import numpy as np

import networkx as nx

from tqdm import tqdm


class TripletDiscovery:

    def __init__(self, model, dataset, temparature):
        self.model = model
        self.dataset = Dataset(dataset)
        self.temperature = temparature

        self.triplets = self.dataset.generate_triplets()
        self.votes_matrix = np.zeros(
            (len(self.dataset.var_description),
            len(self.dataset.var_description))
        )
        self.uncertainty_expert = UncertainExpert(  
            model=self.model,
            temperature=self.temperature
        )
        self.names = self.dataset.var_name
        self.descriptions = self.dataset.var_description 

    async def _triplet_orientation(self):
        for triplet in tqdm(self.triplets):
            reply = await self.uncertainty_expert.triplet_orientation(
                *triplet,
                self.names,
                self.descriptions
            )
            for edge in reply:
                if len(list(edge)) > 1: # not isolated node
                    i = utils.match_var(edge[0], self.names, self.descriptions)
                    j = utils.match_var(edge[1], self.names, self.descriptions)
                    if i != -1 and j != -1 and i != j: 
                        self.votes_matrix[i, j] += 1
            
    def _find_ambiguous_edges(self):
        # unambiguous edges are set to 1
        ambiguous_edges = []
        n = len(self.descriptions)
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge = sorted([self.names[i], self.names[j]])
                    if self.votes_matrix[i, j] == self.votes_matrix[j, i] and edge not in ambiguous_edges:
                        ambiguous_edges.append(edge)
                    elif self.votes_matrix[i, j] > self.votes_matrix[j, i]:
                        self.votes_matrix[i, j] = 1
                        self.votes_matrix[j, i] = 0
                    else: 
                        self.votes_matrix[j, i] = 1
                        self.votes_matrix[i, j] = 0
        return ambiguous_edges
    
    async def _orient_ambiguous_edges(self, ambiguous_edges):
        for edge in ambiguous_edges:
            response = await self.uncertainty_expert.disambiguation(*edge)
            i = utils.match_var(edge[0], self.names, self.descriptions)
            j = utils.match_var(edge[1], self.names, self.descriptions)
            if i != -1 and j != -1 and i != j:
                if response > 1:
                    self.votes_matrix[i, j] = 1
                    self.votes_matrix[j, i] = 0
                else: 
                    self.votes_matrix[j, i] = 1
                    self.votes_matrix[i, j] = 0
            else:
                raise ValueError
        # clean diagonal
        np.fill_diagonal(self.votes_matrix, 0)

    async def search(self):
        await self._triplet_orientation()
        ambiguous_edges = self._find_ambiguous_edges()
        await self._orient_ambiguous_edges(ambiguous_edges)
        G = nx.from_numpy_array(self.votes_matrix, create_using=nx.DiGraph)
        mapping = {i: v for i, v in enumerate(self.dataset.var_description)}
        G = nx.relabel_nodes(G, mapping)

        return G