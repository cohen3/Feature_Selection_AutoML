import os

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

import matplotlib.pyplot as plt
import networkx as nx
import random as rnd


class sub_graph_generator(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")
        self.vertex_threshold = getConfig().eval(self.__class__.__name__, "vertex_threshold")
        self.dataset_table = getConfig().eval(self.__class__.__name__, "dataset_table")
        self.num_of_subgraphs = getConfig().eval(self.__class__.__name__, "num_of_subgraphs_each")
        self.clear_existing_subgraphs = getConfig().eval(self.__class__.__name__, "clear_existing_subgraphs")

    def execute(self, window_start):
        # clear old gpickle graph files
        if self.clear_existing_subgraphs:
            self.clear_graphs()

        # execute random walk
        print('read from full graph:')
        datasets = self.db.execQuery('SELECT DISTINCT dataset_name FROM ' + self.dataset_table)
        print("data sets: {}".format([i[0] for i in datasets]))
        graph_id = 1
        for data in datasets:
            print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKBLUE + 'Dataset: ' + data[0]
                  + bcolors.ENDC + bcolors.ENDC + bcolors.ENDC)
            full_graph = self.db.execQuery(
                'SELECT * FROM ' + self.dataset_table + ' WHERE dataset_name=\'' + data[0] + '\'')
            # calculate number of max vertexes
            max_vertexes = int(len(full_graph) * self.vertex_threshold)

            # generate as many sub graphs as indicated in the config file
            for g in range(self.num_of_subgraphs):
                egdes = {}
                i = 0
                graph = nx.Graph()
                while True:
                    # check vertex threshold or subgraph = full graph
                    if i >= max_vertexes or len(egdes) == len(full_graph):
                        break
                    # choose edge randomly
                    n = rnd.randint(0, len(full_graph) - 1)
                    row = full_graph[n]
                    if n in egdes:
                        continue
                    egdes[n] = True
                    if abs(float(row[4])) >= self.corr_threshold:
                        graph.add_edge(row[2], row[3], weight=float(row[4]))
                        i += 1
                    # TODO: perform the random walk
                    # TODO: dont allow empty graphs
                    # TODO: should we connect non connected nodes with weight 0 o make the graph connected?
                    """
                            for e in nx.non_edges(graph):
                                graph.add_edge(e[0], e[1], weight=0)
                    """
                nx.write_gpickle(graph, 'data/sub_graphs/' + data[0] + '_subgraph' + str(graph_id) + '.gpickle')
                graph_id += 1

    def clear_graphs(self):
        filelist = [f for f in os.listdir('data/sub_graphs') if f.endswith(".gpickle")]
        for f in filelist:
            os.remove(os.path.join('data/sub_graphs', f))

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
