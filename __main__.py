import csv
import os
import sys
import time

from tool_kit.colors import bcolors

from configuration.configuration import getConfig
from dataset_loader.loader import data_loader
from graph_generation.full_graph_generation import graph_generation
from graph_generation.sub_graph_generation_random_selection import random_selection
from graph_generation.sub_graph_generation_random_walk import random_walk
from dataset_generation.graph_feature_extraction import structural_feature_extraction
from dataset_generation.xgboost_dataset_generator import xgboost_generator
from dataset_generation.decision_tree_dataset_generator import Decision_Tree
from regressor.xgboost_regressor import XgboostRegression
from subgraph_embadding.sub2vec import sub2vec
from Test_Module.Test import GA_Feature_Selection
from Test_Module.test_regression_model import test_dataset_cross_validation
from DB.csv_db import CSV_DB
from regressor.RF_Regressor import RandomForestReg
from DB.schema_definition import DB
from graph_generation.sub_graph_generation_algo_feature_selection import algo_feature_selection
from prediction.challenge import challenge_prediction
from feature_selection.SA_feature_selection import simulated_annealing_feature_selection
from Test_Module.benchmark import benchmark

modules_dict = {}

# modules_dict["DB"] = DB
modules_dict["DB"] = CSV_DB
# db = DB()
db = CSV_DB()
modules_dict['data_loader'] = data_loader
modules_dict['graph_generation'] = graph_generation
modules_dict['algo_feature_selection'] = algo_feature_selection
modules_dict['random_selection'] = random_selection
modules_dict['random_walk'] = random_walk
modules_dict['structural_feature_extraction'] = structural_feature_extraction
modules_dict['xgboost_generator'] = xgboost_generator
modules_dict['Decision_Tree'] = Decision_Tree
modules_dict['sub2vec'] = sub2vec
modules_dict['RandomForestReg'] = RandomForestReg
modules_dict['XgboostRegression'] = XgboostRegression
modules_dict['GA_Feature_Selection'] = GA_Feature_Selection
modules_dict['test_dataset_cross_validation'] = test_dataset_cross_validation
modules_dict['challenge_prediction'] = challenge_prediction
modules_dict['simulated_annealing_feature_selection'] = simulated_annealing_feature_selection
modules_dict['benchmark'] = benchmark

window_start = getConfig().eval("DEFAULT", "start_date")
disable_prints = getConfig().eval("DEFAULT", "disable_prints")
if disable_prints:
    sys.stdout = open(os.devnull, 'w')
newbmrk = os.path.isfile("benchmark.csv")
bmrk_file = open("benchmark.csv", 'a', newline='')
bmrk_results = csv.DictWriter(bmrk_file,
                              ["time", "jobnumber", "config", "window_size", "window_start", "dones", "posts",
                               "authors"] + list(modules_dict.keys()),
                              dialect="excel", lineterminator="\n")
if not newbmrk:
    bmrk_results.writeheader()

modules_dict["DB"] = lambda x: x
pipeline = []
for module in getConfig().sections():
    parameters = {}
    if modules_dict.get(module):
        pipeline.append(modules_dict.get(module)(db))

bmrk = {"config": getConfig().getfilename(), "window_start": "setup"}
for module in pipeline:
    print(bcolors.YELLOW + 'Started setup ' + module.__class__.__name__ + bcolors.ENDC)
    T = time.perf_counter()
    module.setUp()
    T = time.perf_counter() - T
    print(bcolors.YELLOW + 'Finished setup ' + module.__class__.__name__ + bcolors.ENDC)
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()

bmrk = {"config": getConfig().getfilename(), "window_start": "execute"}
for module in pipeline:
    T = time.time()
    print(bcolors.YELLOW + 'Started executing ' + module.__class__.__name__ + bcolors.ENDC)

    module.execute(window_start)

    print(bcolors.YELLOW + 'Finished executing ' + module.__class__.__name__ + bcolors.ENDC)
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()
if disable_prints:
    sys.stdout = sys.__stdout__
# x = pipeline[0].execQuery('SELECT * FROM dataset_feature_correlation WHERE feature1=\'f1\'')
# for record in x:
#     print(record)

"""
pipeline:
|--- create database connection
| |--- create schema if needed
|--- dataset loader
| |--- create a correlation matrix
|--- graph generation
| |--- create a table fit to the schema
| |--- fill it with correlations from the matrix
|--- subgraph generation
| |--- randomly choose records from the tables (the full graph)
.
.
.


"""
