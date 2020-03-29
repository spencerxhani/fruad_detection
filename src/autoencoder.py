"""
Reference:https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn import preprocessing
from util import s_to_time_format, string_to_datetime,hour_to_range
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tqdm import tqdm

def value_to_count(df_train, df_test, mode = "train"):

    # continuous_feats = ["locdt","conam","loctm_hour_of_day",
    #                 "loctm_minute_of_hour","loctm_second_of_min"]

    # feats = [f for f in df_test.columns.tolist() if f not in continuous_feats]
    feats = ['acqic', 'bacno', 'cano', 'conam', 'contp', 'csmcu', 'ecfg', 'etymd',
       'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'mcc',
       'mchno', 'ovrlt', 'scity', 'stocn', 'stscd']

    df = pd.concat([df_train[feats], df_test[feats]], axis = 0)
    df_train_ = pd.DataFrame()
    df_test_ = pd.DataFrame()
    for f in tqdm(feats):
        count_dict = df[f].value_counts(dropna = False).to_dict() 
        df_train_[f] = df_train[f].apply(lambda v: count_dict[v])
        df_test_[f] = df_test[f].apply(lambda v: count_dict[v])
    continuous_feats = ['locdt', 'loctm_hour_of_day', 'loctm_minute_of_hour', 'loctm_second_of_min']
    for f in tqdm(continuous_feats):
        df_train_[f] = df_train[f]
        df_test_[f] = df_test[f]
        
    if mode == 'train':
        df_train_["fraud_ind"] = df_train["fraud_ind"]

    return df_train_, df_test_

def feature_normalization_auto(df_train, df_test, mode = "train"):
    """
    return two inputs of autoencoder, one is for train and another one is for test
    """
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
    feats = ['acqic', 'bacno', 'cano', 'conam', 'contp', 'csmcu', 'ecfg', 'etymd',
       'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'locdt', 'mcc',
       'mchno', 'ovrlt', 'scity', 'stocn', 'stscd', 'loctm_hour_of_day',
       'loctm_minute_of_hour', 'loctm_second_of_min']
    scaler = MinMaxScaler()
    df = pd.concat([df_train[feats], df_test[feats]], axis = 0)


    for f in tqdm(feats):
        data = df[f]
        scaler.fit(data)
        df_train[f] = scaler.transform(df_train[f])
        df_test[f] = scaler.transform(df_test[f])
    
    return df_train, df_test

def feature_normalization_auto_v2(df_train, df_test):
    """
    return two inputs of autoencoder, one is for train and another one is for test
    """
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
    feats = ['acqic', 'bacno', 'cano', 'conam', 'contp', 'csmcu', 'ecfg', 'etymd',
       'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'iterm', 'locdt', 'mcc',
       'mchno', 'ovrlt', 'scity', 'stocn', 'stscd', 'loctm_hour_of_day',
       'loctm_minute_of_hour', 'loctm_second_of_min']
    scaler = MinMaxScaler()
    df = pd.concat([df_train[feats], df_test[feats]], axis = 0)



    data = df[feats]
    scaler.fit(data)
    
    if mode == 'train':
        #X_train = df_train[df_train.fraud_ind == 0]
        
        X_train = df_train[feats]
        X_test = df_test[feats]
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    else:
        X_train = scaler.transform(df_train[feats])
        X_test = scaler.transform(df_test[feats])
    
    return X_train, X_test

def autoencoder(input_dim, encoding_dim):
    """
    architecture of autoencoder, we consider this as a dimension reduction method. 
    encoding_dim: int
    input_dim: int.
    """
    from keras.layers import Input, Dense
    from keras.models import Model
    
    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    return autoencoder

def build_model(autoencoder,X_train,X_test,nb_epoch = 100,batch_size = 32):
    """
    X_train: data only including normal transation data

    X_test: data both including fradulant and normal data 
    """
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="../models/autoencoder_v1.h5",
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', 
                                  min_delta=0, 
                                  patience=5, 
                                  verbose=0, 
                                  mode='auto', 
                                  baseline=None, 
                                  restore_best_weights=False)

#     tensorboard = TensorBoard(log_dir='/media/old-tf-hackers-7/logs',
#                               histogram_freq=0,
#                               write_graph=True,
#                               write_images=True)
    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=0,
                        callbacks=[checkpointer, earlystopper]).history
    return history



if __name__ == '__main__':
    df_train = pd.read_csv("/data/yunrui_li/fraud/dataset/train.csv")
    df_test = pd.read_csv("/data/yunrui_li/fraud/dataset/test.csv")


    for df in [df_train, df_test]:
        # pre-processing
        df["loctm_"] = df.loctm.astype(int).astype(str)
        df.loctm_ = df.loctm_.apply(s_to_time_format).apply(string_to_datetime)
        # time-related feature
        df["loctm_hour_of_day"] = df.loctm_.apply(lambda x: x.hour).astype('category')
        df["loctm_minute_of_hour"] = df.loctm_.apply(lambda x: x.minute)
        df["loctm_second_of_min"] = df.loctm_.apply(lambda x: x.second)
        
        # removed the columns no need
        df.drop(columns = ["loctm_", "loctm","txkey"], axis = 1, inplace = True)

    df_train, df_test = value_to_count(df_train, df_test)


    X_train, X_test = feature_preprocessing_auto(df_train, df_test)

    input_dim = X_train.shape[1]
    encoding_dim = 14
    print ('number of raw features', input_dim)
    print ('number of normal data', X_train.shape[0])


    autoencoder = autoencoder(input_dim, encoding_dim)

    print (autoencoder.summary())

    history = build_model(autoencoder,X_train,X_test,nb_epoch = 100, batch_size = 32)
