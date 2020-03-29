import argparse

from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score
import numpy as np
#from pandarallel import pandarallel

from statistics import mean 


def merge_and_split_dfs(df_train, df_test):
    len_train = len(df_train)
    df = pd.concat([df_train, df_test]).reset_index()

    def split_df(df):
        df = df.drop(['index'], axis=1)
        return df.iloc[:len_train], df.iloc[len_train:].drop(['fraud_ind'], axis=1)
    return df, split_df


def get_conam_dict_by_day(df):
    dt_dict = defaultdict(lambda: defaultdict(lambda : 0))
    for index, row in df.iterrows():
        dt_dict[row['cano']][row['locdt']] += row['conam']

    return dt_dict

def _get_last_x_day_conam(cano, locdt, days_back, dt_dict):
    return mean(dict(filter(lambda dt: dt[0]<=locdt and locdt -dt[0] <=days_back, dt_dict.items())).values())

def last_x_day_conam(days_back, df, cano_dict):
    return df[['cano', 'locdt']].apply(lambda row: _get_last_x_day_conam(row['cano'], row['locdt'], days_back, cano_dict[row['cano']]), axis=1)

def main(args):
    df_train = pd.read_csv(args.train_file)
    # loading testing data 
    df_test = pd.read_csv(args.test_file).iloc[:1000]


    y_train = df_train['fraud_ind']
    x_train = df_train.iloc[:1000]


    del df_train

    df, split_df = merge_and_split_dfs(x_train, df_test)

    conam_dict = get_conam_dict_by_day(df)

    print(conam_dict)
    df['last_30_day_mean_conam_per_day'] = last_x_day_conam(30, df, conam_dict)

    df_train, df_test = split_df(df)
    print(df_train.head(5))
    print(df_test.head(5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../../dataset/train.csv', type=str)
    parser.add_argument('--test_file', default='../../dataset/test.csv', type=str)

    main(parser.parse_args())