import os
import random
from sklearn.linear_model import SGDClassifier
import math
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, mutual_info_regression, \
    mutual_info_classif

import warnings


class algo_feature_selection(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        warnings.filterwarnings('ignore')
        self.db = db

    def setUp(self):
        self.clear_existing_subgraphs = getConfig().eval(self.__class__.__name__, "clear_existing_subgraphs")
        self.save_path = getConfig().eval(self.__class__.__name__, "save_path")

    def execute(self, window_start):
        # clear old gpickle graph files
        if os.path.exists('data/sub_graphs'):
            if self.clear_existing_subgraphs:
                print('removing old subgraphs')
                self.clear_graphs()
        else:
            os.mkdir('data/sub_graphs')
        # execute feature selection
        print('read from full graph:')
        datasets = pd.read_csv('data/dataset_out/target_features.csv')['dataset_name'].tolist()
        for data in datasets:
            if not os.path.exists('data/sub_graphs/' + data):
                os.mkdir('data/sub_graphs/' + data)
            print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKBLUE + 'Dataset: ' + data
                  + bcolors.ENDC + bcolors.ENDC + bcolors.ENDC)
            input_df = pd.read_csv('data/dataset_in/' + data.split('_corr_graph')[0] + '.csv')
            input_label = pd.read_csv('data/dataset_out/target_features.csv')
            input_label = input_label.loc[input_label['dataset_name'] == data]
            input_label = input_label[['target_feature']].values.tolist()[0][0]
            label_df = input_df[[input_label]]
            data_df = input_df.loc[:, input_df.columns != input_label]
            counter = 0
            classifiers = {"SGDClassifier": SGDClassifier(alpha=0.1, max_iter=10, shuffle=True,
                                                          random_state=0, tol=None),
                           "LinearSVC": LinearSVC(C=0.01, penalty="l1", dual=False),
                           "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=50),
                           "RandomForestClassifier": RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15),
                           }
            k_best_funcs = {"f_classif": f_classif, "mutual_info_regression": mutual_info_regression,
                            "mutual_info_classif": mutual_info_classif}
            # ################################## Feature Selection Model #####################################
            for clf_name in classifiers.keys():
                counter = 0
                clf = classifiers[clf_name].fit(data_df, label_df)
                model = SelectFromModel(clf, prefit=True)
                feature_idx = model.get_support().tolist()
                indexes = [i for i, x in enumerate(feature_idx) if x is True]
                graph = pd.read_csv('data/dataset_out/' + data + '.csv')
                graph = graph.iloc[indexes, indexes]
                graph = graph.set_index(graph.columns)
                graph = nx.from_pandas_adjacency(graph)
                nx.write_gpickle(graph, 'data/sub_graphs/' + data + '/subgraph_' + clf_name + str(counter) + '.gpickle')
                counter += 1
                for idx in range(int(math.sqrt(len(indexes)) * 2)):
                    feats_to_remove = random.sample(indexes, int(len(indexes) / 3))
                    data_df_cpy = data_df.copy()
                    data_df_cpy.drop(data_df_cpy.columns[feats_to_remove], axis=1, inplace=True)
                    clf = classifiers[clf_name].fit(data_df_cpy, label_df)
                    model = SelectFromModel(clf, prefit=True)
                    feature_idx = model.get_support().tolist()
                    indexes = [i for i, x in enumerate(feature_idx) if x is True]
                    graph = pd.read_csv('data/dataset_out/' + data + '.csv')
                    graph = graph.iloc[indexes, indexes]
                    graph = graph.set_index(graph.columns)
                    graph = nx.from_pandas_adjacency(graph)
                    nx.write_gpickle(graph,
                                     'data/sub_graphs/' + data + '/subgraph_' + clf_name + str(counter) + '.gpickle')
                    counter += 1

            for func_name in k_best_funcs.keys():
                counter = 0
                data_df_cpy = data_df.copy()
                model = SelectKBest(k_best_funcs[func_name], k=int(data_df_cpy.shape[1] / 3))
                model = model.fit(data_df_cpy, label_df)
                feature_idx = model.get_support().tolist()
                indexes = [i for i, x in enumerate(feature_idx) if x is True]
                graph = pd.read_csv('data/dataset_out/' + data + '.csv')
                graph = graph.iloc[indexes, indexes]
                graph = graph.set_index(graph.columns)
                graph = nx.from_pandas_adjacency(graph)
                nx.write_gpickle(graph, 'data/sub_graphs/' + data + '/subgraph' + func_name + str(counter) + '.gpickle')
                counter += 1

            # ################################## End Feature Selection Model #####################################

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
