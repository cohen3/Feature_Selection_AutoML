import os

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random as rnd

from tool_kit.log_utils import get_exclude_list


class random_selection(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")
        self.vertex_threshold = getConfig().eval(self.__class__.__name__, "vertex_threshold")
        # self.dataset_table = getConfig().eval(self.__class__.__name__, "dataset_table")
        self.num_of_subgraphs = getConfig().eval(self.__class__.__name__, "num_of_subgraphs_each")
        self.clear_existing_subgraphs = getConfig().eval(self.__class__.__name__, "clear_existing_subgraphs")
        self.continue_from_log = getConfig().eval(self.__class__.__name__, "continue_from_log")

    def execute(self, window_start):
        print('random selection')
        # clear old gpickle graph files
        if os.path.exists('data/sub_graphs'):
            if self.clear_existing_subgraphs:
                print('removing old subgraphs')
                self.clear_graphs()
        else:
            os.mkdir('data/sub_graphs')
        if self.continue_from_log:
            exclusion_list = get_exclude_list(os.path.join('data', 'loader_log.csv'))
        # execute random walk
        datasets = pd.read_csv('data/dataset_out/target_features.csv')['dataset_name'].tolist()
        graph_id = 1
        for data in datasets:
            if self.continue_from_log and data.replace('_corr_graph', '.csv') in exclusion_list:
                print('skipped: ', data)
                continue
            # if self.db.is_csv:
            #     full_graph = pd.read_csv('data/dataset_out/dataset_feature_correlation.csv')
            #     full_graph = full_graph.loc[full_graph['dataset_name'] == data + ".csv"].values.tolist()
            # else:
            #     full_graph = self.db.execQuery(
            #         'SELECT * FROM ' + self.dataset_table + ' WHERE dataset_name=\'' + data + '\'')
            # calculate number of max vertexes
            df = pd.read_csv('data/dataset_out/' + data + '.csv')
            graph_len = len(df.columns) - 1
            full_graph = nx.read_gpickle('data/full_graphs/' + data.split('_corr_graph')[0] + '_full_graph.gpickle')
            # max_vertexes = int(graph_len * self.vertex_threshold)
            # df = df.set_index(df.columns)
            # full_graph = nx.from_pandas_adjacency(df)
            # egdes_to_remove = [edge for edge in full_graph.edges
            #                    if abs(full_graph[edge[0]][edge[1]]['weight']) > self.corr_threshold]
            # full_graph.remove_edges_from(egdes_to_remove)
            subgraphs_set = []
            # generate as many sub graphs as indicated in the config file
            nodes_count = len(full_graph.nodes) - 1
            g = 0
            # os.mkdir('data/sub_graphs/' + data)
            while g < self.num_of_subgraphs:
                for i in range(int(nodes_count * self.vertex_threshold), nodes_count):
                    if g >= self.num_of_subgraphs:
                        break
                    sub = rnd.sample(list(full_graph.nodes), i)
                    sub = sorted(sub)
                    if sub in subgraphs_set:
                        continue
                    subgraphs_set.append(sub)
                    g1 = full_graph.subgraph(sub)
                    nx.write_gpickle(g1, 'data/sub_graphs/' + data + '_subgraph' + str(graph_id) + '.gpickle')
                    graph_id += 1
                    g += 1
                    # TODO: perform the random walk
                    # TODO: dont allow empty graphs
                    # TODO: should we connect non connected nodes with weight 0 o make the graph connected?

    def clear_graphs(self):
        # filelist = [f for f in os.listdir('data/sub_graphs') if f.endswith(".gpickle")]
        # for f in filelist:
        #     os.remove(os.path.join('data/sub_graphs', f))
        import shutil
        shutil.rmtree("data/sub_graphs")
        os.mkdir("data/sub_graphs")

    def plot_graph(self, graph):
        elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0.8]
        esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 0.8]

        pos = nx.spring_layout(graph)  # positions for all nodes
        # nodes
        nx.draw_networkx_nodes(graph, pos, node_size=20)
        # edges
        nx.draw_networkx_edges(graph, pos, edgelist=elarge,
                               width=1, style='dashed')
        nx.draw_networkx_edges(graph, pos, edgelist=esmall,
                               width=1, alpha=0.5, edge_color='b', style='dashed')
        # labels
        # nx.draw_networkx_labels(graph, pos, font_size=4, font_family='sans-serif')
        plt.axis('off')
        plt.show()
