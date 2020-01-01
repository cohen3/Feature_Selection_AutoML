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
from subgraph_embadding.structural import structural_embedding
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

    def setUp(self):
        if not os.path.exists('data\\walks\\'):
            os.mkdir('data\\walks\\')
        return

    def execute(self, window_start):
        datasets = pd.read_csv('data/dataset_out/target_features.csv')['dataset_name'].tolist()
        with open("data/vectors.csv", "w", newline="") as f:
            pass
        for data in datasets:
            print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKBLUE + 'Dataset: ' + data
                  + bcolors.ENDC + bcolors.ENDC + bcolors.ENDC)
            lst = []
            idx_to_name = {}
            if self.embedding_type is "structural":
                file0 = 'data\\sub_graphs\\' + data + '\\'  # TODO: change to OS
                print("here1")
                lst, idx_to_name = structural_embedding(file0, iterations=self.iterations, dimensions=self.dimensions,
                                                        windowSize=self.windowSize, dm=self.dm,
                                                        walkLength=self.walkLength)
            for key, value in idx_to_name.items():
                print(idx_to_name[key])
                idx_to_name[key] = data + "_" + idx_to_name[key] + ".gpickle"
                print(idx_to_name[key])
            save_vectors(lst, idx_to_name)


def save_vectors(vectors, IdToName):
    results = pd.read_csv('data\\dataset.csv')
    output = open('data\\vectors.csv', 'a+')
    for i in range(len(vectors)):
        score = results.loc[results['graph_name'] == str(IdToName[i])]['macro-avg-f1-score']
        output.write(str(score.tolist()[0]))
        for j in vectors[i]:
            output.write(',' + str(j))
        output.write('\n')
    output.close()
