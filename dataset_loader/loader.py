import os
from os import listdir
from os.path import isfile, join
import pandas as pd

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

class data_loader(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(db)
        self.db = db
        self.csv_data_path = getConfig().eval(self.__class__.__name__, "csv_data_path")
        self.corr_method = getConfig().eval(self.__class__.__name__, "corr_method")

    def setUp(self):
        self.data_files = [f for f in listdir(self.csv_data_path) if isfile(join(self.csv_data_path, f))]

    def execute(self, window_start):

        for file in self.data_files:
            df = pd.read_csv(file)
            columns = df.columns.value
            corr_mat = df.corr(method=self.corr_method)
            dataset_corr_graph = "CREATE TABLE IF NOT EXISTS "+str(os.path.basename)+"_corr_graph (name VARCHAR PRIMARY KEY,"
            for i in range(len(columns)):
                dataset_corr_graph += "feature"+str(i)+" VARCHAR NOT NULL"
            dataset_corr_graph += ");"
            self.db.create_table(dataset_corr_graph)

