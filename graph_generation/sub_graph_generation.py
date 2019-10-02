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

    def execute(self, window_start):
        print('read from full graph:')
        datasets = self.db.execQuery('SELECT DISTINCT dataset_name FROM '+self.dataset_table)
        print("data sets: {}".format([i[0] for i in datasets]))

        for data in datasets:
            full_graph = self.db.execQuery('SELECT * FROM '+self.dataset_table+' WHERE dataset_name=\''+data[0]+'\'')
            max_vertexes = int(len(full_graph)*self.vertex_threshold)
            i = 0
            graph = nx.Graph()
            for j in range(len(full_graph)):
                if i >= max_vertexes:
                    break
                row = full_graph[rnd.randint(0, len(full_graph)-1)]

                if row[4] >= self.corr_threshold:
                    graph.add_edge(row[2], row[3], weight=float(row[4]))
                    i += 1
                # TODO: perform the random walk
                # TODO: dont allow empty graphs
            self.plot_graph(graph)
            self.print_graph_features(graph)
            x = input()

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

    def print_graph_features(self, graph):
        for node in graph.nodes:
            print('Node: ', node)
        for edges in graph.edges:
            print('edges: ', edges)
        try:
            print('diameter: ', nx.diameter(graph))
            print('eccentricity: ', nx.eccentricity(graph))
            print('center: ', nx.center(graph))
            print('periphery: ', nx.periphery(graph))
        except Exception as e:
            print('Graph not connected')
        print('density: ', nx.density(graph))
        print('edges: ', len(graph.edges))
        print('Nodes: ', len(graph.nodes))


