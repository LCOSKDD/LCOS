import copy
import random
import numpy as np  
import pandas as pd
import networkx as nx

from src.utils.settings import TABULAR_DATA_PATH
from src.dataset.dataset import Dataset

class Generator:

    def __init__(self, name: str):
        self.name = name
        self.graph = None
        self.graph = Dataset(name).graph
        
        self.linear_func = [
            lambda m, x, epsilon: np.dot(m, x) + epsilon,
            ]

        self.non_linear_func = [
                lambda m, x, epsilon: np.dot(m, x**2) + epsilon,
                lambda m, xs, epsilon: np.sum([np.cos(x) for x in xs]) + epsilon,
                lambda m, xs, epsilon: np.sum([np.sin(x) for x in xs]) + epsilon,
            ]
        
    def generate(
        self,
        n_samples: int, 
        linear: bool = True,
        noise_distribution = 'normal', 
        filename=None
    ):

        func = self.linear_func if linear else self.non_linear_func

        prime_causes = []
        for node in self.graph.nodes:
            if self.graph.in_degree(node) == 0:
                prime_causes.append(node)

        noise = {}
        for node in self.graph.nodes:
            if noise_distribution == 'normal':
                noise[node] = np.random.normal(scale=0.1, size=(n_samples)) 
            else:
                noise[node] = np.random.uniform(low=-0.1, high=0.1, size=(n_samples))
        dataset = copy.copy(noise)

        for node in self.graph.nodes:
            if self.graph.in_degree(node) > 0:
                in_edges = self.graph.in_edges(node)
                sme = random.choice(func)
                m = np.array(
                    [random.choice([random.uniform(0.1, 1), random.uniform(-1, -0.1)]) 
                     for _ in range(len(in_edges))]
                )
                x = np.array([dataset[edge[0]].ravel() for edge in in_edges])
                epsilon = noise[node].ravel()
                dataset[node] = sme(m, x, epsilon)

        dataset = pd.DataFrame(dataset, columns=self.graph.nodes)

        if filename is not None:
            dataset.to_csv(TABULAR_DATA_PATH + filename, index=False)

        return dataset    


# if __name__ == '__main__':
#     generator = Generator('alarm')
#     generator.generate(1000, linear=False, filename='alarm_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='alarm_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='alarm_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='alarm_linear_uniform.csv')
#     generator = Generator('asia')
#     generator.generate(1000, linear=False, filename='asia_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='asia_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='asia_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='asia_linear_uniform.csv')
#     generator = Generator('cancer')
#     generator.generate(1000, linear=False, filename='cancer_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='cancer_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='cancer_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='cancer_linear_uniform.csv')
#     generator = Generator('child')
#     generator.generate(1000, linear=False, filename='child_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='child_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='child_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='child_linear_uniform.csv')
#     generator = Generator('climate')
#     generator.generate(1000, linear=False, filename='climate_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='climate_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='climate_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='climate_linear_uniform.csv')
#     generator = Generator('covid_1')
#     generator.generate(1000, linear=False, filename='covid_1_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='covid_1_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='covid_1_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='covid_1_linear_uniform.csv')
#     generator = Generator('covid_2')
#     generator.generate(1000, linear=False, filename='covid_2_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='covid_2_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='covid_2_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='covid_2_linear_uniform.csv')
#     generator = Generator('covid_3')
#     generator.generate(1000, linear=False, filename='covid_3_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='covid_3_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='covid_3_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='covid_3_linear_uniform.csv')
#     generator = Generator('covid_4')
#     generator.generate(1000, linear=False, filename='covid_4_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='covid_4_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='covid_4_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='covid_4_linear_uniform.csv')
#     generator = Generator('genetic')
#     generator.generate(1000, linear=False, filename='genetic_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='genetic_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='genetic_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='genetic_linear_uniform.csv')
#     generator = Generator('insurance')
#     generator.generate(1000, linear=False, filename='insurance_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='insurance_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='insurance_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='insurance_linear_uniform.csv')
#     generator = Generator('msu')
#     generator.generate(1000, linear=False, filename='msu_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='msu_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='msu_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='msu_linear_uniform.csv')
#     generator = Generator('neighborhood')
#     generator.generate(1000, linear=False, filename='neighborhood_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='neighborhood_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='neighborhood_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='neighborhood_linear_uniform.csv')
#     generator = Generator('opioids')
#     generator.generate(1000, linear=False, filename='opioids_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='opioids_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='opioids_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='opioids_linear_uniform.csv')
#     generator = Generator('sachs')
#     generator.generate(1000, linear=False, filename='sachs_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='sachs_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='sachs_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='sachs_linear_uniform.csv')
#     generator = Generator('supermarket')
#     generator.generate(1000, linear=False, filename='supermarket_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='supermarket_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='supermarket_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='supermarket_linear_uniform.csv')
#     generator = Generator('sachs')
#     generator.generate(1000, linear=False, filename='sachs_non_linear_normal.csv')
#     generator.generate(1000, linear=True, filename='sachs_linear_normal.csv')
#     generator.generate(1000, linear=False, noise_distribution='uniform', filename='sachs_non_linear_uniform.csv')
#     generator.generate(1000, linear=True, noise_distribution='uniform', filename='sachs_linear_uniform.csv')
    
