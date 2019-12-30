import itertools
import os
import random

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.clique import *
from networkx.algorithms.traversal import *
import random as rnd


class random_walk(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.method = getConfig().eval(self.__class__.__name__, "method")
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")



    def execute(self, window_start):
        # execute random walk
        print('read from full graph:')
        datasets = pd.read_csv('data/dataset_out/target_features.csv')['dataset_name'].tolist()
        print("data sets: {}".format([i for i in datasets]))
        graph_id = 1
        for data in datasets:
            print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKBLUE + 'Dataset: ' + data
                  + bcolors.ENDC + bcolors.ENDC + bcolors.ENDC)
            # calculate number of max vertexes
            df = pd.read_csv('data/dataset_out/'+data+'.csv')
            df = df.set_index(df.columns)
            print('building full graph...')
            full_graph = nx.from_pandas_adjacency(df)
            print('calculating invalid edges...')
            egdes_to_remove = [edge for edge in full_graph.edges
                               if full_graph[edge[0]][edge[1]]['weight'] < self.corr_threshold]
            print('removing edges...')
            full_graph.remove_edges_from(egdes_to_remove)
            if self.method == 'clique' or self.method == 'all':
                print('finding cliques...')
                cliques = nx.find_cliques(full_graph)
                self.__save_subgraphs(cliques, data, 'clique')
            if self.method == 'tree' or self.method == 'all':
                print('building trees...')
                sources = random.choices(list(full_graph.nodes), k=len(list(full_graph.nodes))*0.1)
                for node in sources:
                    tree = dfs_tree(full_graph, source=node, depth_limit=int(len(full_graph)/2))
                    self.__save_subgraphs([tree], data, 'tree')


    def __save_subgraphs(self, graph_iter, name, kind):
        graph_id = 1
        try:
            for sub in graph_iter:
                sub_graph = self.complete_graph_from_list(list(sub))
                nx.write_gpickle(sub_graph, 'data/sub_graphs/' + name + '_subgraph_' + kind + str(graph_id) + '.gpickle')
                graph_id += 1
        except MemoryError as me:
            pass

    def complete_graph_from_list(self, L, create_using=None):
        G = nx.empty_graph(len(L), create_using)
        if len(L) > 1:
            edges = itertools.combinations(L, 2)
            G.add_edges_from(edges)
        return G

    def clear_graphs(self):
        filelist = [f for f in os.listdir('data/sub_graphs') if f.endswith(".gpickle")]
        for f in filelist:
            os.remove(os.path.join('data/sub_graphs', f))

