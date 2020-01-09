import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product


def perason(df):
    return df.corr(method="pearson")


def kendell(df):
    return df.corr(method="kendall")


def cos_sim(df):
    # print(df.columns[0])
    # df = df.drop(df.columns[0], axis=1)
    names = list(df.columns)
    # names = names[1:]
    df = df.transpose()
    sim_array = cosine_similarity(df)
    sim_df = pd.DataFrame(data=sim_array[0:, 0:],    # values
                          index=names,    # 1st column as index
                          columns=names)  # 1st row as the column names
    # cols = list(df.columns)
    # df_corr = pd.DataFrame(index=cols, columns=cols)
    # for x in list(product(cols, repeat=2)):
    #     df_corr[x[0]][x[1]] = (1.0 - cosine(df[x[0]], df[x[1]]))
    # # print(df_corr)
    # return df_corr
    return sim_df
