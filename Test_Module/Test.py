import csv
import pickle
import time

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors
import pandas as pd
import random


class GA_Feature_Selection(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.iter = 15

    def setUp(self):
        print('setting up test_module')
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.target = getConfig().eval(self.__class__.__name__, "target")
        self.xgb_model_loaded = pickle.load(open('data/RF_regression_model.dat', "rb"))
        population = pd.read_csv(self.dataset).columns
        self.source_inds = list(population)
        self.population = list()
        for ind in population:
            l = list()
            for ind2 in population:
                n = int(len(population)/10)
                l.append(random.sample(self.source_inds, n))
            self.population.append(l)

    def execute(self, window_start):
        mutation_chance = len(self.population) / 2
        crossover_chance = len(self.population) / 10
        population_size = len(self.population)
        for i in range(self.iter):
            scores = list()
            for ind in self.population:
                m = random.randint(0, len(self.population))
                c = random.randint(0, len(self.population))
                if m <= mutation_chance:
                    new_feature = random.sample(self.source_inds, 1)
                    old_feature = random.sample(ind, 1)
                    if new_feature not in ind:
                        if len(ind) > 1:
                            ind.remove(*old_feature)
                        ind.append(*new_feature)
                if c <= crossover_chance:
                    ind2 = random.sample(self.population, 1)
                    size = min(len(ind), len(ind2))
                    if size != 1:
                        cxpoint = random.randint(1, size - 1)
                        ind[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind[cxpoint:]

            # fitness time
            for ind in self.population:
                scores.append((self.__fitness(ind), ind))

            max = sorted(scores, key=lambda tup: tup[0], reverse=True)[:50]
            self.population = [i[1] for i in max]
            for j in range(population_size - 50):
                ind = random.sample(self.population, 1)[0]
                new_ind = [f for f in ind]
                new_feature = random.sample(self.source_inds, 1)
                old_feature = random.sample(ind, 1)
                if new_feature not in ind:
                    new_ind.remove(*old_feature)
                    new_ind.append(*new_feature)
                self.population.append(new_ind)
        print(self.population[0][0])


    def cleanUp(self, window_start):
        print('clean up test_module')

    def __fitness(self, individuals):
        df = pd.read_csv(self.dataset)
        print(df.head(10))
        x=input()
        # X = df[individuals]
        # return self.xgb_model_loaded.predict(X)
        return len(individuals)
