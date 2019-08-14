import csv
import configparser

class csv_read():
    def __init__(self,file_name):
        self.file_name = file_name
        config = configparser.ConfigParser()
    def execute(self):
        self.read_csv