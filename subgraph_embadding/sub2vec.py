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
        embaddings_list = []
        with open("data/vectors.csv", "w", newline="") as f:
            pass
        for data in datasets:
            print(bcolors.BOLD + bcolors.UNDERLINE + bcolors.OKBLUE + 'Dataset: ' + data
                  + bcolors.ENDC + bcolors.ENDC + bcolors.ENDC)
            lst\
                = []
            idx_to_name = {}
            if self.embedding_type is "structural":
                file0 = 'data\\sub_graphs\\' + data + '\\'
                print("here1")
                lst, idx_to_name = structural_embedding(file0, iterations=self.iterations, dimensions=self.dimensions,
                                                        windowSize=self.windowSize, dm=self.dm,
                                                        walkLength=self.walkLength)

            saveVectors(lst, idx_to_name)
        #         print(lst[0])
        #         for idx in idx_to_name:
        #             lst2[idx].append(idx_to_name[idx])
        #         input(lst2[0])
        #         embaddings_list.append(lst2)
        # with open("data/vectors.csv", "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(embaddings_list)

    # parser = argparse.ArgumentParser(description="sub2vec.")
    # parser.add_argument('--input', nargs='?', default='input', required=True, help='Input directory')
    # parser.add_argument('--property', default='n', choices=['n', 's'], required=True,
    #                     help='Type of subgraph property to presernve. For neighborhood property add " --property n"'
    #                          ' and for the structural property " --property s" ')
    # parser.add_argument('--walkLength', default=100000, type=int, help='length of random walk on each subgraph')
    # parser.add_argument('--output', required=True, default='output', help='Output representation file')
    # parser.add_argument('--d', default=128, type=int, help='dimension of learned feautures for each subgraph.')
    # parser.add_argument('--iter', default=20, type=int, help='training iterations')
    # parser.add_argument('--windowSize', default=2, type=int,
    #                     help='Window size of the model.')
    # parser.add_argument('--p', default=0.5, type=float,
    #                     help='meta parameter.')
    # parser.add_argument('--model', default='dm', choices=['dbon', 'dm'],
    #                     help='models for learninig vectors SV-DM (dm) or SV-DBON (dbon).')
    # args = parser.parse_args()
    # if args.property == 's':
    #     structural_embedding(args)
    # else:
    #     neighborhood_embedding(args)


# if __name__ == '__main__':
#     main()
def saveVectors(vectors, IdToName):
    output = open('data\\vectors.csv', 'a+')
    for i in range(len(vectors)):
        output.write(str(IdToName[i]))
        for j in vectors[i]:
            output.write(','+str(j))
        output.write('\n')
    output.close()
