import sys
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from contextlib import contextmanager
import gc 
from util import s_to_time_format, string_to_datetime, hour_to_range, kfold_lightgbm, kfold_xgb
from util import _time_elapsed_between_last_transactions,time_elapsed_between_last_transactions
#from util import add_auto_encoder_feature
from time import strftime, localtime
import logging
import sys
from config import Configs
from extraction import merge_and_split_dfs, get_conam_dict_by_day, last_x_day_conam
from sklearn.metrics import f1_score

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
#log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
log_file = '../result/{}_tuning.log'.format(strftime("%y%m%d-%H%M", localtime()))
logger.addHandler(logging.FileHandler(log_file))

def lgb_f1_score(y_pred, y_true):
    """evaluation metric"""
    #print ("y_pred",y_pred)
    #print ("y_true",y_true)
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true.get_label(), y_hat), True

def bayes_parameter_opt_lgb(X, y, 
                            init_round=15, 
                            opt_round=25, 
                            n_folds=5, 
                            random_seed=1030,
                            n_estimators=10000,
                            learning_rate=0.05, 
                            output_process=True):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature='auto', free_raw_data = False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction,
                 max_depth, lambda_l1, lambda_l2, min_split_gain, 
                 min_child_weight):
        params = {'application':'binary',
                  'num_iterations': n_estimators, 
                  'learning_rate':learning_rate, 
                  'early_stopping_round':100, 
                  }
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, 
                           train_data, 
                           nfold=n_folds,
                           seed=random_seed, 
                           stratified=True, 
                           categorical_feature = "auto",
                           feval=lgb_f1_score)
        print (cv_result)
        return max(cv_result['f1-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: 
        pd.DataFrame(opt_params.res).sort_values(by = "target", ascending=False).to_csv("../result/bayes_opt_result.csv")
    return lgbBO.max["target"], lgbBO.max["params"] # best score and best parameter

def group_target_by_cols(df_train, df_test, recipe):
    df = pd.concat([df_train, df_test], axis = 0)
    for m in range(len(recipe)):
        cols = recipe[m][0]
        for n in range(len(recipe[m][1])):
            target = recipe[m][1][n][0]
            method = recipe[m][1][n][1]
            name_grouped_target = method+"_"+target+'_BY_'+'_'.join(cols)
            tmp = df[cols + [target]].groupby(cols).agg(method)
            tmp = tmp.reset_index().rename(index=str, columns={target: name_grouped_target})
            df_train = df_train.merge(tmp, how='left', on=cols)
            df_test = df_test.merge(tmp, how='left', on=cols)
    del tmp
    gc.collect()
    
    return df_train, df_test

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def main(args):
    with timer("Process train/test application"):
        #-------------------------
        # load dataset
        #-------------------------
        df_train = pd.read_csv(args.train_file)
        df_test = pd.read_csv(args.test_file)

        #-------------------------
        # pre-processing
        #-------------------------

        for cat in Configs.CATEGORY:
            df_train[cat] = df_train[cat].astype('category') #.cat.codes
            df_test[cat] = df_test[cat].astype('category')
            
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
        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add bacno/cano feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.CONAM_AGG_RECIPE_1)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add iterm-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.ITERM_AGG_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add conam-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.CONAM_AGG_RECIPE_2)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add hour-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.HOUR_AGG_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add cano/conam feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.CANO_CONAM_COUNT_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add cano/bacno latent feature"):
        df = pd.read_csv("../features/bacno_latent_features.csv")
        df_train = df_train.merge(df, on = "bacno", how = "left")
        df_test = df_test.merge(df, on = "bacno", how = "left")
        df = pd.read_csv("../features/cano_latent_features.csv")
        df_train = df_train.merge(df, on = "cano", how = "left")
        df_test = df_test.merge(df, on = "cano", how = "left")

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add locdt-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.LOCDT_CONAM_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add mchno-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.MCHNO_CONAM_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add scity-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.SCITY_CONAM_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add stocn-related feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.STOCN_CONAM_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add mchno/bacno latent feature"):
        df = pd.read_csv("../features/bacno_latent_features_w_mchno.csv")
        df_train = df_train.merge(df, on = "bacno", how = "left")
        df_test = df_test.merge(df, on = "bacno", how = "left")
        df = pd.read_csv("../features/mchno_latent_features.csv")
        df_train = df_train.merge(df, on = "mchno", how = "left")
        df_test = df_test.merge(df, on = "mchno", how = "left")

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add elapsed time feature"):
        df = pd.concat([df_train, df_test], axis = 0)
        df.sort_values(by = ["bacno","locdt"], inplace = True)
        
        df["time_elapsed_between_last_transactions"] = df[["bacno","locdt"]] \
        .groupby("bacno").apply(_time_elapsed_between_last_transactions).values
        
        df_train = df[~df.fraud_ind.isnull()]
        df_test = df[df.fraud_ind.isnull()]
        
        df_test.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
        del df
        gc.collect()

        df_train["time_elapsed_between_last_transactions"] = df_train[["bacno","locdt","time_elapsed_between_last_transactions"]] \
        .groupby(["bacno","locdt"]).apply(time_elapsed_between_last_transactions).values
        
        df_test["time_elapsed_between_last_transactions"] = df_test[["bacno","locdt","time_elapsed_between_last_transactions"]] \
        .groupby(["bacno","locdt"]).apply(time_elapsed_between_last_transactions).values
        
        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add elapsed time aggregate feature"):
        df_train, df_test = group_target_by_cols(df_train, df_test, Configs.TIME_ELAPSED_AGG_RECIPE)

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))  

    feats = [f for f in df_train.columns if f not in ["fraud_ind"]]
    X,y = df_train[feats], df_train.fraud_ind
    return X,y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../../dataset/train.csv', type=str)
    parser.add_argument('--test_file', default='../../dataset/test.csv', type=str)

    X,y = main(parser.parse_args())

    opt_score, opt_params = bayes_parameter_opt_lgb(X, y, 
                                         init_round=5, 
                                         opt_round=10, 
                                         n_folds=2, 
                                         random_seed=6, 
                                         n_estimators=10, 
                                         learning_rate=0.2)
