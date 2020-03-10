import csv
import os
import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors
import pandas as pd
import random


class test_dataset_cross_validation(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.data_path = getConfig().eval(self.__class__.__name__, "data")

    def execute(self, window_start):
        location = os.path.join('data', 'sub_graphs')
        datasets = next(os.walk(location))[1]
        full_dataset = pd.read_csv(self.data_path)
        output_file = os.path.join('data', 'dataset_cross_validation.csv')
        i = 1
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_number', 'R-squared_score', 'test_dataset'])
            writer.writeheader()
            for test_dataset in datasets:
                filtered_df = full_dataset[full_dataset['dataset_name'] != test_dataset]
                test_df = full_dataset[full_dataset['dataset_name'] == test_dataset]

                filtered_df = filtered_df.drop('dataset_name', axis=1)
                test_df = test_df.drop('dataset_name', axis=1)

                X_train = filtered_df.drop('target', axis=1)
                y_train = filtered_df['target']

                X_test = test_df.drop('target', axis=1)
                y_test = test_df['target']

                rfc = RandomForestRegressor(n_jobs=-1, random_state=22)
                model = rfc.fit(X_train, y_train)

                pred1 = model.score(X_test, y_test)
                writer.writerow({'run_number': i, 'R-squared_score': pred1,
                                 'test_dataset': test_dataset.split('_corr_graph')[0]})
                print('{}: Score = {}\tTest dataset = {}'.format('run_'+str(i), pred1, test_dataset))
                i += 1


