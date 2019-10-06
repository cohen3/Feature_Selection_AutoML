import time
from os import walk, listdir
from os.path import isfile

import xgboost as xgb
import networkx as nx
import matplotlib.pyplot as plt
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class xgboost_generator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.max_depth = getConfig().eval(self.__class__.__name__, "max_depth")
        self.eta = getConfig().eval(self.__class__.__name__, "eta")
        self.silent = getConfig().eval(self.__class__.__name__, "silent")
        self.objective = getConfig().eval(self.__class__.__name__, "objective")
        self.nthread = getConfig().eval(self.__class__.__name__, "nthread")
        self.epochs = getConfig().eval(self.__class__.__name__, "epochs")
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.exclude_table_list = getConfig().eval(self.__class__.__name__, "exclude_table_list")

    def execute(self, window_start):

        for file in listdir('data/sub_graphs'):
            print('\n'+file)
            dataset = file.split("_corr_graph")[0]
            print(bcolors.OKBLUE+"Dataset: "+dataset+bcolors.ENDC)
            graph = nx.read_gpickle('data/sub_graphs/'+file)
            self.print_graph_features(graph)
            self.plot_graph(graph)

    def __fit(self, target_name):
        train_df = None  # TODO: replace this with actual pandas DataFrame
        test_df = None  # TODO: replace this with actual pandas DataFrame
        train_df = train_df.na.fill(0)

        X_train = train_df.drop(target_name, axis=1)
        Y_train = train_df[target_name]

        dtrain = xgb.DMatrix(X_train, label=Y_train)
        param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': self.silent, 'objective': self.objective,
                 'nthread': self.nthread, 'eval_metric': ['auc', 'aucpr']}  # old parameters

        evallist = [(dtrain, 'train')]

        start = time.perf_counter()
        bst = xgb.train(param, dtrain, self.epochs, evallist)
        end = time.perf_counter()

        test_df = test_df.na.fill(0)
        pred_df = bst.predict(test_df)
        print(pred_df.show())
        return {'acc': 0, 'aucpr': 0, 'train_time': (end - start)}

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
        # for node in graph.nodes:
        #     print('Node: ', node)
        # for edges in graph.edges:
        #     print('edges: ', edges)
        # for e in nx.non_edges(graph):
        #     print(e)
        try:
            print('diameter: ', nx.diameter(graph))
            print('eccentricity: ', nx.eccentricity(graph))
            print('center: ', nx.center(graph))
            print('periphery: ', nx.periphery(graph))
        except Exception as e:
            print('Graph not connected')
        print('density: ', nx.density(graph))
        # print('degree: ', nx.degree(graph))
        print('Average degree: ', sum([i[1] for i in nx.degree(graph)]) / len(nx.degree(graph)))
        print('edges: ', len(graph.edges))
        print('Nodes: ', len(graph.nodes))
        print('self loops: ', len(list(nx.nodes_with_selfloops(graph))))
        print('edges to nodes ratio: ', len(graph.nodes) / len(graph.edges))
        print('negative edges: ', nx.is_negatively_weighted(graph))