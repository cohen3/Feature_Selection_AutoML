# coding=utf-8
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import networkx as nx
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors
import dataset_loader.corr_calc
from tool_kit.log_utils import get_exclude_list

class data_loader(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db
        self.continue_from_log = getConfig().eval(self.__class__.__name__, "continue_from_log")
        self.csv_data_path = getConfig().eval(self.__class__.__name__, "csv_data_path")
        self.corr_method = getConfig().eval(self.__class__.__name__, "corr_function")
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")
        self.data_files = [f for f in listdir(self.csv_data_path) if isfile(join(self.csv_data_path, f))]
        self.corr_mat = {}
        self.targets = {"dataset_name": [], "target_feature": []}

    def setUp(self):
        if self.continue_from_log:
            exclusion_list = get_exclude_list(os.path.join('data', 'loader_log.csv'))
        for file in self.data_files:
            if self.continue_from_log and file in exclusion_list:
                print('skipped: ', file)
                continue
            df = pd.read_csv(join(self.csv_data_path, file))
            df = self.preprocess(df)
            # print(os.path.splitext(file)[0] + '_corr_graph\ntarget:' + df.columns[-1])
            print('dataset name: {}\tfeatures: {}\tinstances: {}'.format(os.path.splitext(file)[0],
                                                                         df.shape[1],
                                                                         df.shape[0]))
            method_to_call = getattr(dataset_loader.corr_calc, self.corr_method)
            if 'class' in df.columns:
                features_df = df.drop('class', axis=1)
            elif 'target' in df.columns:
                features_df = df.drop('target', axis=1)
            else:
                features_df = df.drop(df.columns[-1], axis=1)

            # creates a full graph (corr matrix)
            try:
                self.corr_mat[str(os.path.splitext(file)[0])+'_corr_graph'] = method_to_call(features_df)
            except Exception as e:
                print(e)
                print("is nan: ".format(np.where(np.isnan(df))))
                continue
            self.targets["dataset_name"].append(str(os.path.splitext(file)[0])+'_corr_graph')
            if 'target' in df.columns:
                self.targets["target_feature"].append('target')
            elif 'class' in df.columns:
                self.targets["target_feature"].append('class')
            else:
                self.targets["target_feature"].append(df.columns[-1])
            fg = self.corr_mat[str(os.path.splitext(file)[0])+'_corr_graph'].set_index(self.corr_mat[str(os.path.splitext(file)[0])+'_corr_graph'].columns)
            full_graph = nx.from_pandas_adjacency(fg)
            if not os.path.exists(os.path.join('data', 'full_graphs')):
                os.makedirs(os.path.join('data', 'full_graphs'))
            egdes_to_remove = [edge for edge in full_graph.edges
                               if abs(full_graph[edge[0]][edge[1]]['weight']) > self.corr_threshold]
            full_graph.remove_edges_from(egdes_to_remove)
            bb = nx.betweenness_centrality(full_graph)
            nx.set_node_attributes(full_graph, bb, 'global_betweenness')
            dg = {k: v for k, v in full_graph.degree()}
            nx.set_node_attributes(full_graph, dg, 'global_degree')
            average_weights = dict()
            for node in full_graph.nodes:
                av = np.average([full_graph[node][n]['weight'] for n in full_graph.neighbors(node)])
                average_weights[node] = av
            nx.set_node_attributes(full_graph, average_weights, 'global_average_edge_weight')
            h, a = nx.hits(full_graph, max_iter=1000)
            nx.set_node_attributes(full_graph, a, 'global_authority')
            nx.set_node_attributes(full_graph, h, 'global_hub')
            nx.write_gpickle(full_graph, os.path.join('data', 'full_graphs',
                                                      str(os.path.splitext(file)[0] + '_full_graph.gpickle')))
            # builtin {‘pearson’, ‘kendall’, ‘spearman’}

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        for name, file_corr_mat in self.corr_mat.items():
            # each is a full graph (corr matrix)
            print('saving '+name)
            self.db.df_to_table(df=file_corr_mat, name=name, mode='replace')
        df = pd.DataFrame(self.targets)
        self.db.df_to_table(df=df, name="target_features", mode='replace')


