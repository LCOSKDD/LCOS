from typing import List

import yaml
import pickle
import pandas as pd
from networkx.drawing.nx_agraph import write_dot, read_dot
import networkx as nx

from src.utils.settings import TEXT_DATA_PATH, TABULAR_DATA_PATH, DATA_PATH, GRAPH_PATH


class Dataset:
    """ Abstract class for dataset """
    def __init__(self, name: str, filename: str = None, linear: bool = True, normal: bool = True) -> None:
        self.name = name
        self.filename = filename if filename is not None else name + '.csv'
        self.df_text = pd.read_csv(TEXT_DATA_PATH + self.filename)
        linear = 'linear' if linear else 'non_linear'
        normal = 'normal' if normal else 'uniform'
        self.data = pd.read_csv(f'{TABULAR_DATA_PATH}{self.name}_{linear}_{normal}.csv')
        self.graph = pickle.load(open(f'{GRAPH_PATH}{self.name}.pkl', 'rb'))

    def __getitem__(self, key):
        return self.df_text[key].values
    
    def __len__(self):
        return len(self.df_text)
    
    def var_description_lang(self, index: int, language: str = 'english') -> str:
        return self.df_text[f'var_description_{language}'][index]
    
    def to_dot(self):
        write_dot(self.graph, f'{GRAPH_PATH}{self.name}.dot')

    def generate_triplets(self):
        # generate all triplets from variables description 
        triplets = []
        for var_i in self.var_name:
            for var_j in self.var_name:
                for var_k in self.var_name:
                    if (var_i != var_j and var_i != var_k and var_j != var_k
                        and sorted([var_i, var_j, var_k]) not in triplets):

                        triplets.append(sorted([var_i, var_j, var_k]))

        return triplets

    @property
    def var_description(self) -> List[str]:
        return self.df_text.var_description_english.values
    
    @property
    def var_name(self) -> List[str]:
        return self.df_text.var_name
