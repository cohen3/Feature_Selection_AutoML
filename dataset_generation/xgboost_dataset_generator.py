import time
from os import walk, listdir
from os.path import isfile

import xgboost as xgb
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as ps
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import math

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class xgboost_generator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db
        self.num_class = {}

    def setUp(self):
        self.max_depth = getConfig().eval(self.__class__.__name__, "max_depth")
        self.eta = getConfig().eval(self.__class__.__name__, "eta")
        self.silent = getConfig().eval(self.__class__.__name__, "silent")
        self.objective_multi = getConfig().eval(self.__class__.__name__, "objective_multi")
        self.objective_binary = getConfig().eval(self.__class__.__name__, "objective_binary")
        self.nthread = getConfig().eval(self.__class__.__name__, "nthread")
        self.epochs = getConfig().eval(self.__class__.__name__, "epochs")
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.exclude_table_list = getConfig().eval(self.__class__.__name__, "exclude_table_list")

    def execute(self, window_start):

        for file in listdir('data/sub_graphs'):
            print('\n' + file)
            dataset = file.split("_corr_graph")[0]
            print(bcolors.OKBLUE + "Dataset: " + dataset + bcolors.ENDC)
            graph = nx.read_gpickle('data/sub_graphs/' + file)
            graph_features = self.print_graph_features(graph)
            # self.plot_graph(graph)
            X_train, X_test, y_train, y_test, num = self.prepare_dataset(dataset, graph)
            res = self.__fit(X_train, X_test, y_train, y_test, self.num_class[dataset])
            self.commit_results(graph_features, res)

    def prepare_dataset(self, dataset_name='', graph=None):

        # extracting feature names from graph
        features = [node for node in graph.nodes]

        # reading the dataset into pandas df
        df = pd.read_csv('data/dataset_in/' + dataset_name + ".csv")

        # get the dataset's target
        target_feature = self.db.execQuery("SELECT target_feature FROM target_features WHERE dataset_name=\'"
                                           + dataset_name + "_corr_graph\'")[0][0]

        # convert categorical to numeric
        if df[target_feature].dtype != 'float64' and df[target_feature].dtype != 'int64':
            df[target_feature] = df[target_feature].astype('category').cat.codes

        # split data to test and train
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_feature],
                                                            test_size=0.15, random_state=2)
        # read number of classes into dictionary
        if dataset_name not in self.num_class:
            q1 = "SELECT COUNT(DISTINCT "+target_feature+") as num FROM df "
            q2 = "SELECT DISTINCT "+target_feature+" as classes FROM df "
            num = ps.sqldf(q1, locals())
            classes = ps.sqldf(q2, locals())
            self.num_class[dataset_name] = int(num.iloc[0]['num'])
            self.labels = ['{}'.format(i) for i in list(classes['classes'].values)]

        return X_train, X_test, y_train, y_test, self.num_class[dataset_name]

    def __fit(self, X_train, X_test, Y_train, y_test, num):
        # distinction between multi class to binary
        objective = self.objective_multi
        if num <= 2:
            objective = self.objective_binary
            num -= 1

        # prepare data
        dtrain = xgb.DMatrix(X_train, label=Y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': self.silent, 'objective': objective,
                 'nthread': self.nthread, 'num_class': num}  # old parameters
        evallist = [(dtrain, 'train')]

        # training
        start = time.perf_counter()
        print(bcolors.GREY)
        bst = xgb.train(param, dtrain, self.epochs, evallist)
        print(bcolors.ENDC)
        end = time.perf_counter()

        # record performances
        # if num + 1 <= 2:
        #     res = self.get_performances(bst, dtest, y_test, y_test.values)
        # else:
        #     res = self.get_performances_multiclass(bst, dtest, y_test, None)
        res = self.get_performances_multiclass(bst, dtest, y_test, (num + 1 <= 2))
        print("Training time: {:.3f} sec".format(end-start))
        res['train_time'] = '{:.3f}'.format(end - start)
        return res

    def get_performances(self, model, X_test, Y, labels):
        # TODO: make compatible with multiclass
        res = {}
        preds = model.predict(X_test)

        check2 = preds.round()
        score = precision_score(labels, check2)
        print('precision score: {:.6f}'.format(score))
        res['precision'] = '{:.6f}'.format(score)

        score = recall_score(labels, check2)
        print('recall score: {:.6f}'.format(score))
        res['recall'] = '{:.6f}'.format(score)

        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        print('roc_auc: {:.6f}'.format(roc_auc))
        res['roc_auc'] = '{:.6f}'.format(roc_auc)

        predictions = []
        for value in preds:
            if value > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        accuracy = accuracy_score(Y, predictions)
        print("Accuracy: {:.6f}".format(accuracy))
        res['Accuracy'] = '{:.6f}'.format(accuracy)
        return res

    def get_performances_multiclass(self, model, X_test, y_test, binary):

        res = {}
        y_pred = model.predict(X_test)
        y_pred = [float(i) for i in list(y_pred)]
        y_test_list = [i for i in list(y_test.values)]
        if binary:
            y_pred = [math.floor(i) if i < 0.5 else 1 for i in y_pred]
        res = classification_report(y_test_list, y_pred, target_names=self.labels, output_dict=True)
        return res

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

        # print('density: ', res['density'])
        # print('Average degree: ', res['Avg_degree'])
        # print('Average weight: ', res['Avg_weight'])
        # print('edges: ', len(graph.edges))
        # print('Nodes: ', len(graph.nodes))
        # print('self loops: ', res['self_loops'])
        # print('edges to nodes ratio: ', res['edge_to_node_ratio'])
        # print('negative edges: ', res['negative_edges'])
        # nodes = [node for node in graph.nodes]
        # print(nodes)

        return res

    def commit_results(self, graph_features, performance):
        print(bcolors.UNDERLINE + bcolors.OKBLUE + 'Features:' + bcolors.ENDC + bcolors.ENDC)
        for key, value in graph_features.items():
            print('{:20}{}'.format(key, value))
        print(bcolors.UNDERLINE + bcolors.RED + 'Target:' + bcolors.ENDC + bcolors.ENDC)
        for key, value in performance.items():
            print('{:20}{}'.format(key, value))
