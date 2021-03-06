import argparse
import networkx as nx
from os import listdir
import os
import pandas as pd
import csv
from tool_kit.colors import bcolors
import numpy as np
from os.path import isfile, join
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from subgraph_embadding.structural import structural_embedding, generate_walk_file
from subgraph_embadding.neighborhood import neighborhood_embedding


# nx.write_gpickle(g1, 'data/sub_graphs/' + data + '_subgraph' + str(graph_id) + '.gpickle')
class sub2vec(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db
        self.iterations = getConfig().eval(self.__class__.__name__, "iterations")
        self.dimensions = getConfig().eval(self.__class__.__name__, "dimensions")
        self.windowSize = getConfig().eval(self.__class__.__name__, "windowSize")
        self.dm = getConfig().eval(self.__class__.__name__, "dm")
        self.walkLength = getConfig().eval(self.__class__.__name__, "walkLength")
        self.embedding_type = getConfig().eval(self.__class__.__name__, "embedding_type")
        self.att = getConfig().eval(self.__class__.__name__, "attribute")

    def setUp(self):
        dir = os.path.join('data', 'walks', '')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return

    def execute(self, window_start):
        datasets = pd.read_csv('data/dataset_out/target_features.csv')['dataset_name'].tolist()
        with open("data/vectors.csv", "w", newline="") as f:
            f.write('target')
            f.write(',dataset_name')
            for feature in range(self.dimensions):
                f.write(',' + 'col_' + str(feature))
            f.write('\n')
        idx_to_name = {}
        file0 = ""
        counter = 0
        for data in datasets:
            file0 = os.path.join('data', 'sub_graphs')
            idx_to_name_temp, counter = generate_walk_file(counter, file0, self.walkLength, 0.5)
            idx_to_name = {**idx_to_name, **idx_to_name_temp}
        # print(idx_to_name)
        #
        # input()
        lst = structural_embedding(file0, iterations=self.iterations, dimensions=self.dimensions,
                                   windowSize=self.windowSize, dm=self.dm,
                                   walkLength=self.walkLength)

        for key, value in idx_to_name.items():
            # print('1')
            # print(idx_to_name[key])

            idx_to_name[key] = idx_to_name[key] + ".gpickle"
            # print('2')
            # print(idx_to_name[key])
        save_vectors(lst, idx_to_name, self.att)


def save_vectors(vectors, IdToName, att):
    data = os.path.join('data', 'dataset.csv')
    vectors_dir = os.path.join('data', 'vectors.csv')
    results = pd.read_csv(data)
    output = open(vectors_dir, 'a+')
    for i in range(len(vectors)):
        data_name = str(IdToName[i]).split('_corr')[0]
        score = results.loc[results['graph_name'] == str(IdToName[i])][att]
        output.write(str(score.tolist()[0]))
        output.write(',')
        output.write(data_name)
        for j in vectors[i]:
            output.write(',' + str(j))
        output.write('\n')
    output.close()

# TODO: train sub2vec over all the graphs together
# TODO: graph convolution network can generate embedding for a given graph
# TODO: implement K-fold validation
# TODO: after K-Fold, benchmanrk the sub2vec
