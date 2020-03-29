"""
the Matrix Fatorization-based features are created in a unsupervised way 
and therefore lower the need of expert knowledge for the creation 
of the fraud detection system.

bacno-mchno matrix
bacno-contp matrix
bacno-scity matrix
mchno-loctm_hour_of_day matrix
cano-mchno matrix
"""
import os
import gc
import argparse
import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy import sparse
from util import s_to_time_format, string_to_datetime, hour_to_range

def check_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return True

def main(args):
    #---------------------------------
    # load dataset
    #---------------------------------
    df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)
    for df in [df_train, df_test]:
        # pre-processing
        df["loctm_"] = df.loctm.astype(int).astype(str)
        df.loctm_ = df.loctm_.apply(s_to_time_format).apply(string_to_datetime)
        # # time-related feature
        df["loctm_hour_of_day"] = df.loctm_.apply(lambda x: x.hour).astype('category')
        df["loctm_minute_of_hour"] = df.loctm_.apply(lambda x: x.minute)
        df["loctm_second_of_min"] = df.loctm_.apply(lambda x: x.second)
        # df["loctm_absolute_time"] = [h*60+m for h,m in zip(df.loctm_hour_of_day,df.loctm_minute_of_hour)]
        df["hour_range"] = df.loctm_.apply(lambda x: hour_to_range(x.hour)).astype("category")
        # removed the columns no need
        df.drop(columns = ["loctm_"], axis = 1, inplace = True)
        # auxiliary fields
        df["day_hr_min"] = ["{}:{}:{}".format(i,j,k) for i,j,k in zip(df.locdt,df.loctm_hour_of_day,df.loctm_minute_of_hour)]
        df["day_hr_min_sec"] = ["{}:{}:{}:{}".format(i,j,k,z) for i,j,k,z in zip(df.locdt,df.loctm_hour_of_day,df.loctm_minute_of_hour,df.loctm_second_of_min)]

    df = pd.concat([df_train, df_test], axis = 0)

    del df_train, df_test
    gc.collect()
    #---------------------------------
    # prepare bacno-cano count matrix
    #---------------------------------
    ls = [args.row_name, args.column_name]
    interactions = df[ls+["loctm"]].groupby(ls).count().reset_index().rename(columns = {"loctm":"num_count"})
    # min-max normalization
    max_ = interactions.num_count.max()
    min_ = interactions.num_count.min()
    interactions.num_count = interactions.num_count.apply(lambda x : 1.0 * (x-min_)/(max_-min_))

    # row
    num_bacno = interactions[args.row_name].nunique()
    bacno_dict = {e:i for i, e in enumerate(interactions[args.row_name].unique())}
    bacno_dict_inv = {e:i for i,e in bacno_dict.items()}
    # column
    num_cano = interactions[args.column_name].nunique()
    cano_dict = {e:i for i, e in enumerate(interactions[args.column_name].unique())}
    cano_dict_inv = {e:i for i,e in cano_dict.items()}

    data = np.zeros(shape = (num_bacno,num_cano), dtype = np.float32)
    for ix, row in interactions.iterrows():
        bacno_index = bacno_dict[row[args.row_name]] # row
        cano_index = cano_dict[row[args.column_name]] # column
        data[bacno_index,cano_index] = row.num_count
    data = sparse.csr_matrix(data)

    del interactions
    gc.collect()

    print ("=" *100)
    print ("data preprocessing done")
    #---------------------------------
    # modeling
    #---------------------------------
    no_components = 10
    # Instantiate and train the model
    model = LightFM(loss='logistic',no_components=no_components)
    model.fit(interactions = data,
              epochs=100, 
              num_threads=2,
              verbose = True)
    #---------------------------------
    # saving
    #---------------------------------
    check_folder(args.latent_feature_path)
    # item_embeddings
    df = pd.concat(
        [pd.DataFrame({args.column_name:list(cano_dict_inv.values())}),
         pd.DataFrame(model.item_embeddings,columns = ["{}_{}_latent_features_{}".format(args.row_name,args.column_name,i) for i in range(no_components)])
        ],axis = 1)
    df.to_csv(os.path.join(args.latent_feature_path, "{}_{}_latent_features.csv".format(args.row_name,args.column_name)), index = False)
    # user_embeddings
    df = pd.concat(
        [pd.DataFrame({args.row_name:list(bacno_dict_inv.values())}),
         pd.DataFrame(model.user_embeddings,columns = ["{}_latent_features_{}_w_{}".format(args.row_name,i,args.column_name) for i in range(no_components)])
        ],axis = 1)
    df.to_csv(os.path.join(args.latent_feature_path, "{}_latent_features_w_{}.csv".format(args.row_name,args.column_name)), index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../../dataset/train.csv', type=str)
    parser.add_argument('--test_file', default='../../dataset/test.csv', type=str)
    parser.add_argument('--latent_feature_path', default='../features/', type=str)
    parser.add_argument('--column_name', default='cano', type=str, help = "cano, mchno, contp, scity, stocn, loctm_hour_of_day")
    parser.add_argument('--row_name', default='bacno', type=str, help = "cano, contp, scity, stocn")

    main(parser.parse_args())
