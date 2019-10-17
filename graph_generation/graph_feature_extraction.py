import csv
import time

import networkx as nx
import networkx.algorithms.approximation as aprox
import networkx.algorithms.components as conn
import networkx.algorithms.clique as clq
import os

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors
from tool_kit.graph_features import features


class graph_feature_extraction(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.path = getConfig().eval(self.__class__.__name__, "path")

    def execute(self, window_start):
        graph_dataset = open("data/dataset_out/new_dataset.csv", 'a', newline='')
        graph_dataset_file = csv.DictWriter(graph_dataset, fieldnames=features, dialect="excel", lineterminator="\n")
        graph_dataset_file.writeheader()
        for file in os.listdir(self.path):
            dataset = file.split("_corr_graph")[0]
            print(bcolors.OKBLUE + "Dataset: " + dataset + bcolors.ENDC)
            graph = nx.read_gpickle('data/sub_graphs/' + file)
            res = self.extract_simple_features(graph)
            res['graph'] = file

            # TODO: find more algorithms to extract features from
            """
                a list of algorithms can be found here:
                https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.html
            """
            # algorithms = ['max_clique', 'node_connectivity', 'average_clustering']
            # algorithms = ['number_connected_components']
            algorithms = ['graph_clique_number', 'graph_number_of_cliques']
            for algorithm in algorithms:
                self.run_algo(clq, algorithm, graph)
            x=input()

    def run_algo(self, module, method_name, graph):
        s = time.perf_counter()
        ret = getattr(module, method_name)(graph)
        print(method_name+': ', ret)
        e = time.perf_counter() - s
        print(str(e) + ' sec\n')

    def extract_simple_features(self, graph):
        res = {}
        try:
            print('diameter: ', nx.diameter(graph))
            print('eccentricity: ', nx.eccentricity(graph))
            print('center: ', nx.center(graph))
            print('periphery: ', nx.periphery(graph))
            res['connected'] = True
        except Exception as e:
            print('Graph not connected')
            res['connected'] = False

        res['density'] = '{:.6f}'.format(nx.density(graph))
        res['Avg_degree'] = '{:.6f}'.format(sum([i[1] for i in nx.degree(graph)]) / len(nx.degree(graph)))
        res['Avg_weight'] = '{:.6f}'.format(sum([graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]) / len(
            [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]))
        res['edges'] = len(graph.edges)
        res['nodes'] = len(graph.nodes)
        res['self_loops'] = len(list(nx.nodes_with_selfloops(graph)))
        res['edge_to_node_ratio'] = '{:.6f}'.format(len(graph.nodes) / len(graph.edges))
        res['negative_edges'] = nx.is_negatively_weighted(graph)
        return res

