import pandas as pd
from scipy.spatial.distance import cosine
from itertools import product


def perason(df):
    return df.corr(method="pearson")


def kendell(df):
    return df.corr(method="kendall")


def cos_sim(df):
    cols = list(df.columns)
    df_corr = pd.DataFrame(index=cols, columns=cols)
    for x in list(product(cols, repeat=2)):
        df_corr[x[0]][x[1]] = (1.0 - cosine(df[x[0]], df[x[1]]))
    print(df_corr)
    return df_corr
