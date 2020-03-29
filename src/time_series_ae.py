import numpy as np
import pandas as pd
import sys
from util import s_to_time_format, string_to_datetime,hour_to_range
from tqdm import tqdm

def value_to_count(df_train, df_test, df_train_normal_cano_id, df_):
    """
    convert categorial features into number of occurence in the dataset.
    """
    # separate continuous feature and categorial features
    feats = ['acqic', 'bacno', 'cano', 'conam', 'contp', 'csmcu', 'ecfg', 'etymd',
       'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'locdt',
       'mcc', 'mchno', 'ovrlt', 'scity', 'stocn', 'stscd', 'loctm_hour_of_day',
       'loctm_minute_of_hour', 'loctm_second_of_min'] 
    cont_feats = [
                  'conam',
                  'iterm', 
                  'locdt',
                  'loctm_hour_of_day',
                  'loctm_minute_of_hour', 
                  'loctm_second_of_min']
    feats = [f for f in feats if f not in cont_feats]
    # we only coner categorial features
    
    df = pd.concat([df_train[feats], df_test[feats]], axis = 0)
    for f in tqdm(feats):
        count_dict = df[f].value_counts(dropna = False).to_dict() 
        df_train_normal_cano_id[f] = df_train_normal_cano_id[f].apply(lambda v: count_dict[v])
        df_train[f] = df_train[f].apply(lambda v: count_dict[v])
        df_test[f] = df_test[f].apply(lambda v: count_dict[v])
        df_[f] = df_[f].apply(lambda v: count_dict[v])
    return df_train,df_test,df_train_normal_cano_id, df_

def feature_normalization_auto(df_train, df_test, df_train_normal_cano_id,df_):
    """
    return two inputs of autoencoder, one is for train and another one is for test
    """
    #from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
    feats = ['acqic', 'bacno', 'cano', 'conam', 'contp', 'csmcu', 'ecfg', 'etymd',
       'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'locdt',
       'mcc', 'mchno', 'ovrlt', 'scity', 'stocn', 'stscd', 'loctm_hour_of_day',
       'loctm_minute_of_hour', 'loctm_second_of_min']
    df = pd.concat([df_train[feats], df_test[feats]], axis = 0)


    for f in tqdm(feats):
        try:
            #scaler = MinMaxScaler()
            max_ = df[f].max()
            min_ = df[f].min()
            df_train_normal_cano_id[f] = df_train_normal_cano_id[f].apply(lambda x: (x-min_)/(max_-min_))
            df_[f] = df_[f].apply(lambda x: (x-min_)/(max_-min_))
            #df_test[f] = df_test[f].apply(lambda x: (x-min_)/(max_-min_))
        except:
            print(f)
    return df_train_normal_cano_id,df_

def partition_(df, num_features):
    data = []
    for i in range(len(df)):
        out = None
        if i == 0:
            out = np.concatenate(((np.zeros((2,num_features))),df.iloc[:1].values))
        elif i== 1:
            out = np.concatenate(((np.zeros((1,num_features))),df.iloc[:i+1].values))
        else:
            out = df.iloc[i+1-3:i+1].values
        data.append(out)
    return data

def partition(df_, sequence_length = 3):
    feats = [f for f in df_.columns if f not in {"fraud_ind","cano_help","locdt_help"}]
    sequences = []
    for _, df in df_.groupby(by = "cano_help"):
        data = partition_(df[feats], num_features = len(feats))
        for d in data:
            sequences.append(d)
    return sequences

def get_sequence_dataframe(df):
    df_train_sequences = partition(df)
    df_train_sequences = np.concatenate(df_train_sequences)
    df_train_sequences = pd.DataFrame(df_train_sequences)
    return df_train_sequences


if __name__ == '__main__':
    #-----------------------------
    # load data
    #-----------------------------
    df_train = pd.read_csv("/data/yunrui_li/fraud/dataset/train.csv")
    df_test = pd.read_csv("/data/yunrui_li/fraud/dataset/test.csv")


    for df in [df_train, df_test]:
        # pre-processing
        df["loctm_"] = df.loctm.astype(int).astype(str)
        df.loctm_ = df.loctm_.apply(s_to_time_format).apply(string_to_datetime)
        # time-related feature
        df["loctm_hour_of_day"] = df.loctm_.apply(lambda x: x.hour)
        df["loctm_minute_of_hour"] = df.loctm_.apply(lambda x: x.minute)
        df["loctm_second_of_min"] = df.loctm_.apply(lambda x: x.second)

        # removed the columns no need
        df.drop(columns = ["loctm_", "loctm","txkey"], axis = 1, inplace = True)

    df_train["cano_locdt_index"] = ["{}_{}_{}_{}_{}".format(str(i),str(j),str(k),str(l),str(m)) for i,j,k,l,m in zip(df_train.cano,
                                                                                       df_train.locdt,
                                                                                       df_train.loctm_hour_of_day,
                                                                                       df_train.loctm_minute_of_hour,
                                                                                       df_train.loctm_second_of_min,
                                                                                      )]
    df_test["cano_locdt_index"] = ["{}_{}_{}_{}_{}".format(str(i),str(j),str(k),str(l),str(m)) for i,j,k,l,m in zip(df_test.cano,
                                                                                      df_test.locdt,
                                                                                      df_test.loctm_hour_of_day,
                                                                                      df_test.loctm_minute_of_hour,
                                                                                      df_test.loctm_second_of_min,
                                                                                     )]

    df_train["cano_help"] = df_train.cano
    df_test["cano_help"] = df_test.cano

    df_train["locdt_help"] = df_train.locdt
    df_test["locdt_help"] = df_test.locdt

    df_train["loctm_hour_of_day_help"] = df_train.loctm_hour_of_day
    df_test["loctm_hour_of_day_help"] = df_test.loctm_hour_of_day

    df_train["loctm_minute_of_hour_help"] = df_train.loctm_minute_of_hour
    df_test["loctm_minute_of_hour_help"] = df_test.loctm_minute_of_hour

    df_train["loctm_second_of_min_help"] = df_train.loctm_second_of_min
    df_test["loctm_second_of_min_help"] = df_test.loctm_second_of_min

    #-----------------------------
    # feature extraction
    #-----------------------------
    df = pd.concat([df_train, df_test], axis = 0)
    df.sort_values(by = ["cano", "locdt","loctm_hour_of_day","loctm_minute_of_hour","loctm_second_of_min"], inplace = True)

    #-----------------------------
    # prepare training data
    #-----------------------------
    df_train.sort_values(by = ["cano", "locdt","loctm_hour_of_day","loctm_minute_of_hour","loctm_second_of_min"], inplace = True)

    # df_train, df_test = value_to_count(df_train, df_test)
    # df_train, df_test = feature_normalization_auto(df_train, df_test)

    fraud_cano_id = df_train[df_train.fraud_ind == 1].cano.unique().tolist()

    df_train_normal_cano_id = df_train[~df_train.cano.isin(fraud_cano_id)]
    print ("number of training data",df_train_normal_cano_id.shape)

    df_train, df_test, df_train_normal_cano_id, df = value_to_count(df_train, df_test,df_train_normal_cano_id, df)
    df_train_normal_cano_id, df = feature_normalization_auto(df_train, df_test,df_train_normal_cano_id, df)

    #-----------------------------
    # post-processing
    #-----------------------------
    df.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
    df_train_normal_cano_id.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
    feats = ['acqic', 'bacno', 'cano', 'conam', 'contp', 'csmcu', 'ecfg', 'etymd',
       'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'locdt',
       'mcc', 'mchno', 'ovrlt', 'scity', 'stocn', 'stscd', 'loctm_hour_of_day',
       'loctm_minute_of_hour', 'loctm_second_of_min'] + ["cano_locdt_index","cano_help","locdt_help"]

    df = df[feats]
    df_train_normal_cano_id = df_train_normal_cano_id[feats]

    #-----------------------------
    # get train/test data
    #-----------------------------

    X_train = get_sequence_dataframe(df_train_normal_cano_id)
    Feature = get_sequence_dataframe(df)
    #-----------------------------
    # modeling (unsupervised learning)
    #-----------------------------
    import sys
    #sys.path.append("/data/yunrui_li/fraud/DeepADoTS")
    #from src.algorithms.dagmm import DAGMM
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"  
    from DAGMM import DAGMM
    detectors = DAGMM(num_epochs=50, sequence_length=3)
    detectors.fit(X_train.iloc[:,:-1].copy())

    score = detectors.predict(Feature.iloc[:,:-1].copy())
    output = pd.DataFrame({"cano_locdt_index":Feature.iloc[:,-1]})
    output["score"] = score

    print (output.shape)

    output["cosine_errors_mean"] = detectors.prediction_details["cosine_errors_mean"]
    output["euclidean_errors_mean"]  = detectors.prediction_details["euclidean_errors_mean"]
    data = detectors.prediction_details["reconstructions_mean"]
    reconstructions_mean = pd.DataFrame(data.T,
                 columns = ["reconstructions_mean_latent_features_{}".format(i) for i in range(data.shape[0])]
                )
    print (reconstructions_mean.shape)
    data = detectors.prediction_details["latent_representations"]
    latent_representations = pd.DataFrame(data.T,
                 columns = ["latent_representations_latent_features_{}".format(i) for i in range(data.shape[0])]
                )
    print (latent_representations.shape)
    output = pd.concat([output,reconstructions_mean,latent_representations], axis = 1)
    print (output.shape)

    feature = []
    for i in range(len(output)):
        if i%3 == 2:
            feature.append(output.iloc[i:i+1])
    feature = pd.concat(feature,axis = 0)

    feature.to_csv("/data/yunrui_li/fraud/fraud_detection/features/DAGMM_features_modified_2.csv", index = False)
