"""
python3 main.py ../../dataset/train.csv ../../dataset/test.csv ../result/cv_results.csv ../result/submission.csv > ../result/logs.txt

make train

"""
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from contextlib import contextmanager
import gc 
from util import s_to_time_format, string_to_datetime, hour_to_range, kfold_lightgbm, kfold_xgb
from util import rolling_stats_target_by_cols
#from util import _time_elapsed_between_last_transactions,time_elapsed_between_last_transactions
#from util import num_transaction_in_past_n_days
#from util import add_auto_encoder_feature
#from util import group_target_by_cols_split_by_users
from time import strftime, localtime
import logging
import sys
from config import Configs

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
#log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
log_file = '../result/{}.log'.format(strftime("%y%m%d-%H%M", localtime()))
logger.addHandler(logging.FileHandler(log_file))

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

        # reduced memory    
        del tmp
        gc.collect()
    
    return df_train, df_test

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def main(args):
    if args.load_feature == True:
        with timer("Load train/test features extracted"):
            #-------------------------
            # load dataset
            #-------------------------
            df_train = pd.read_csv("../features/train.csv")
            df_test = pd.read_csv("../features/test.csv")

            #-------------------------
            # pre-processing
            #-------------------------

            for cat in Configs.CATEGORY:
                df_train[cat] = df_train[cat].astype('category') #.cat.codes
                df_test[cat] = df_test[cat].astype('category')
            for df in [df_train, df_test]:
                df["hour_range"] = df["hour_range"].astype('category')

            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))     

    else:
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
                # auxiliary fields
                df["day_hr_min"] = ["{}:{}:{}".format(i,j,k) for i,j,k in zip(df.locdt,df.loctm_hour_of_day,df.loctm_minute_of_hour)]
                df["day_hr_min_sec"] = ["{}:{}:{}:{}".format(i,j,k,z) for i,j,k,z in zip(df.locdt,df.loctm_hour_of_day,df.loctm_minute_of_hour,df.loctm_second_of_min)]

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
            df = pd.read_csv("../features/bacno_latent_features_w_cano.csv")
            df_train = df_train.merge(df, on = "bacno", how = "left")
            df_test = df_test.merge(df, on = "bacno", how = "left")
            df = pd.read_csv("../features/bacno_cano_latent_features.csv")
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
            df = pd.read_csv("../features/bacno_mchno_latent_features.csv")
            df_train = df_train.merge(df, on = "mchno", how = "left")
            df_test = df_test.merge(df, on = "mchno", how = "left")

            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add time second-level feature on bacno"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.HOUR_AGG_SEC_LEVEL_RECIPE_BACNO,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add time second-level feature on cano"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.HOUR_AGG_SEC_LEVEL_RECIPE_CANO,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add time second-level feature on mchno"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.HOUR_AGG_SEC_LEVEL_RECIPE_MCHNO,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add time second-level feature on csmcu/stocn/scity"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.HOUR_AGG_SEC_LEVEL_RECIPE,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add time second-level feature on acqic/csmcu/stocn/scity"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.HOUR_AGG_SEC_LEVEL_RECIPE_2,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add conam-related feature v3"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.CONAM_AGG_RECIPE_3,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add locdt-related feature v2"):
            df_train, df_test = group_target_by_cols(df_train, df_test, Configs.LOCDT_CONAM_RECIPE_2)

            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add conam-related feature v4"):
            df_train, df_test = group_target_by_cols(
                df_train, 
                df_test, 
                Configs.CONAM_AGG_RECIPE_4,
                )
            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add cano/mchno latent feature"):
            df = pd.read_csv("../features/cano_latent_features_w_mchno.csv")
            df_train = df_train.merge(df, on = "cano", how = "left")
            df_test = df_test.merge(df, on = "cano", how = "left")
            df = pd.read_csv("../features/cano_mchno_latent_features.csv")
            df_train = df_train.merge(df, on = "mchno", how = "left")
            df_test = df_test.merge(df, on = "mchno", how = "left")

            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add cano/locdt latent feature"):
            df = pd.read_csv("../features/cano_latent_features_w_locdt.csv")
            df_train = df_train.merge(df, on = "cano", how = "left")
            df_test = df_test.merge(df, on = "cano", how = "left")
            df = pd.read_csv("../features/cano_locdt_latent_features.csv")
            df_train = df_train.merge(df, on = "locdt", how = "left")
            df_test = df_test.merge(df, on = "locdt", how = "left")

            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

        with timer("Add mchno/locdt latent feature"):
            df = pd.read_csv("../features/mchno_latent_features_w_locdt.csv")
            df_train = df_train.merge(df, on = "mchno", how = "left")
            df_test = df_test.merge(df, on = "mchno", how = "left")
            df = pd.read_csv("../features/mchno_locdt_latent_features.csv")
            df_train = df_train.merge(df, on = "locdt", how = "left")
            df_test = df_test.merge(df, on = "locdt", how = "left")

            logger.info("Train application df shape: {}".format(df_train.shape))
            logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add mchno time aggregate average feature"):
        # df = pd.read_csv("../features/average_mchno_time_agg.csv")
        # df_train = df_train.merge(df, on = "txkey", how = "left")
        # df_test = df_test.merge(df, on = "txkey", how = "left")
        df = pd.read_csv("../features/average_mchno_mean_conam_in_past_7_days.csv")
        df_train = df_train.merge(df, on = "mchno", how = "left")
        df_test = df_test.merge(df, on = "mchno", how = "left")

        df = pd.read_csv("../features/average_mchno_mean_conam_in_past_14_days.csv")
        df_train = df_train.merge(df, on = "mchno", how = "left")
        df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_std_conam_in_past_7_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_std_conam_in_past_14_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_min_conam_in_past_7_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_min_conam_in_past_14_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_max_conam_in_past_7_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_max_conam_in_past_14_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_median_conam_in_past_7_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        # df = pd.read_csv("../features/average_mchno_median_conam_in_past_14_days.csv")
        # df_train = df_train.merge(df, on = "mchno", how = "left")
        # df_test = df_test.merge(df, on = "mchno", how = "left")

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    with timer("Add bacno time aggregate average feature"):
        # df = pd.read_csv("../features/average_bacno_min_conam_in_past_7_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_max_conam_in_past_7_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        df = pd.read_csv("../features/average_bacno_mean_conam_in_past_7_days.csv").iloc[:,1:]
        df_train = df_train.merge(df, on = "bacno", how = "left")
        df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_median_conam_in_past_7_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_std_conam_in_past_7_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_min_conam_in_past_14_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_max_conam_in_past_14_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        df = pd.read_csv("../features/average_bacno_mean_conam_in_past_14_days.csv").iloc[:,1:]
        df_train = df_train.merge(df, on = "bacno", how = "left")
        df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_median_conam_in_past_14_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        # df = pd.read_csv("../features/average_bacno_std_conam_in_past_14_days.csv").iloc[:,1:]
        # df_train = df_train.merge(df, on = "bacno", how = "left")
        # df_test = df_test.merge(df, on = "bacno", how = "left")

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add lstm latent features"):
    #     # df = pd.read_csv("../features/average_bacno_min_conam_in_past_7_days.csv").iloc[:,1:]
    #     # df_train = df_train.merge(df, on = "bacno", how = "left")
    #     # df_test = df_test.merge(df, on = "bacno", how = "left")

    #     # df = pd.read_csv("../features/average_bacno_max_conam_in_past_7_days.csv").iloc[:,1:]
    #     # df_train = df_train.merge(df, on = "bacno", how = "left")
    #     # df_test = df_test.merge(df, on = "bacno", how = "left")

    #     df = pd.read_csv("../features/lstm_features.csv")
    #     df_train = df_train.merge(df, on = "txkey", how = "left")
    #     df_test = df_test.merge(df, on = "txkey", how = "left")

    # with timer("Add mcc time aggregate average feature"):
    #     df = pd.read_csv("../features/average_mcc_median_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = "mcc", how = "left")
    #     df_test = df_test.merge(df, on = "mcc", how = "left")

    #     df = pd.read_csv("../features/average_mcc_max_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = "mcc", how = "left")
    #     df_test = df_test.merge(df, on = "mcc", how = "left")

    #     df = pd.read_csv("../features/average_mcc_min_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = "mcc", how = "left")
    #     df_test = df_test.merge(df, on = "mcc", how = "left")

    #     df = pd.read_csv("../features/average_mcc_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = "mcc", how = "left")
    #     df_test = df_test.merge(df, on = "mcc", how = "left")

    #     df = pd.read_csv("../features/average_mcc_std_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = "mcc", how = "left")
    #     df_test = df_test.merge(df, on = "mcc", how = "left")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add scity time aggregate feature"):
    #     df = pd.read_csv("../features/scity_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["scity","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["scity","locdt"], how = "left")

    #     df = pd.read_csv("../features/scity_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["scity","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["scity","locdt"], how = "left")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add stocn time aggregate feature"):
    #     df = pd.read_csv("../features/stocn_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["stocn","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["stocn","locdt"], how = "left")

    #     df = pd.read_csv("../features/stocn_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["stocn","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["stocn","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add acqic time aggregate feature"):
    #     df = pd.read_csv("../features/acqic_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["acqic","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["acqic","locdt"], how = "left")

    #     df = pd.read_csv("../features/acqic_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["acqic","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["acqic","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add mchno time aggregate feature"):
    #     df = pd.read_csv("../features/mchno_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["mchno","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["mchno","locdt"], how = "left")

    #     df = pd.read_csv("../features/mchno_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["mchno","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["mchno","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add bacno time aggregate feature"):
    #     df = pd.read_csv("../features/bacno_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["bacno","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["bacno","locdt"], how = "left")

    #     df = pd.read_csv("../features/bacno_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["bacno","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["bacno","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add mcc time aggregate feature"):
    #     df = pd.read_csv("../features/mcc_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["mcc","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["mcc","locdt"], how = "left")

    #     df = pd.read_csv("../features/mcc_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["mcc","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["mcc","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add hcefg time aggregate feature"):
    #     df = pd.read_csv("../features/hcefg_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["hcefg","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["hcefg","locdt"], how = "left")

    #     df = pd.read_csv("../features/hcefg_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["hcefg","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["hcefg","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add contp time aggregate feature"):
    #     df = pd.read_csv("../features/contp_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["contp","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["contp","locdt"], how = "left")

    #     df = pd.read_csv("../features/contp_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["contp","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["contp","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add etymd time aggregate feature"):
    #     df = pd.read_csv("../features/etymd_mean_conam_in_past_7_days.csv")
    #     df_train = df_train.merge(df, on = ["etymd","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["etymd","locdt"], how = "left")

    #     df = pd.read_csv("../features/etymd_mean_conam_in_past_14_days.csv")
    #     df_train = df_train.merge(df, on = ["etymd","locdt"], how = "left")
    #     df_test = df_test.merge(df, on = ["etymd","locdt"], how = "left")
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add cano/bacno ratio feature"):
    #     from util import num_transaction

    #     df_train = num_transaction(df_train,target = "cano")
    #     df_test = num_transaction(df_test,target = "cano")
    #     df_train = num_transaction(df_train,target = "bacno")
    #     df_test = num_transaction(df_test,target = "bacno")

    #     df_train["cano_ratio"] = df_train["cano_len"] / df_train["bacno_len"]
    #     df_test["cano_ratio"] = df_test["cano_len"] / df_test["bacno_len"]

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add if_conam_zero feature"):
    #     from util import if_conam_zero

    #     df_train = if_conam_zero(df_train)
    #     df_test = if_conam_zero(df_test)

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add DAGMM latent feature"):
    #     df_train["cano_locdt_index"] = ["{}_{}".format(str(i),str(j)) for i,j in zip(df_train.cano,df_train.locdt)]
    #     df_test["cano_locdt_index"] = ["{}_{}".format(str(i),str(j)) for i,j in zip(df_test.cano,df_test.locdt)]

    #     df = pd.read_csv("../features/DAGMM_features_less_input.csv")
    #     df_train = df_train.merge(df, on = "cano_locdt_index", how = "left").drop_duplicates("txkey")
    #     df_test = df_test.merge(df, on = "cano_locdt_index", how = "left").drop_duplicates("txkey")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))
    #     del df
    #     gc.collect()

    # with timer("Add bacno/locdt latent feature"):
    #     df = pd.read_csv("../features/bacno_latent_features_w_locdt.csv")
    #     df_train = df_train.merge(df, on = "bacno", how = "left")
    #     df_test = df_test.merge(df, on = "bacno", how = "left")
    #     df = pd.read_csv("../features/bacno_locdt_latent_features.csv")
    #     df_train = df_train.merge(df, on = "locdt", how = "left")
    #     df_test = df_test.merge(df, on = "locdt", how = "left")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add stocn/locdt latent feature"):
    #     df = pd.read_csv("../features/stocn_latent_features_w_locdt.csv")
    #     df_train = df_train.merge(df, on = "stocn", how = "left")
    #     df_test = df_test.merge(df, on = "stocn", how = "left")
    #     df = pd.read_csv("../features/stocn_locdt_latent_features.csv")
    #     df_train = df_train.merge(df, on = "locdt", how = "left")
    #     df_test = df_test.merge(df, on = "locdt", how = "left")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add elapsed time feature"):
    #     df = pd.concat([df_train, df_test], axis = 0)
    #     df.sort_values(by = ["bacno","locdt"], inplace = True)
        
    #     df["time_elapsed_between_last_transactions"] = df[["bacno","locdt"]] \
    #     .groupby("bacno").apply(_time_elapsed_between_last_transactions).values
        
    #     df_train = df[~df.fraud_ind.isnull()]
    #     df_test = df[df.fraud_ind.isnull()]
        
    #     df_test.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
    #     del df
    #     gc.collect()

    #     df_train["time_elapsed_between_last_transactions"] = df_train[["bacno","locdt","time_elapsed_between_last_transactions"]] \
    #     .groupby(["bacno","locdt"]).apply(time_elapsed_between_last_transactions).values
        
    #     df_test["time_elapsed_between_last_transactions"] = df_test[["bacno","locdt","time_elapsed_between_last_transactions"]] \
    #     .groupby(["bacno","locdt"]).apply(time_elapsed_between_last_transactions).values
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add elapsed time aggregate feature"):
    #     df_train, df_test = group_target_by_cols(df_train, df_test, Configs.TIME_ELAPSED_AGG_RECIPE)

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))  

    # with timer("Add elapsed time related feature"):
    #     df_train, df_test = group_target_by_cols(df_train, df_test, Configs.TIME_ELAPSED_AGG_RECIPE_2)

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))  

    # with timer("Add historical-related feature"):
    #     df = pd.concat([df_train, df_test], axis = 0)
    #     df.sort_values(by = ["bacno","locdt"], inplace = True)
        
    #     for past_n_days in [2,3,4,5,6,7,14,30]:
    #         df["num_transaction_in_past_{}_days".format(past_n_days)] = df[["bacno","locdt"]].groupby("bacno")\
    #         .apply(lambda x: num_transaction_in_past_n_days(x,past_n_days)).values

    #     df_train = df[~df.fraud_ind.isnull()]
    #     df_test = df[df.fraud_ind.isnull()]
        
    #     df_test.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
    #     del df
    #     gc.collect()

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))  

    # with timer("Add descriptive stats in past transactions feature"):
    #     df_train, df_test = rolling_stats_target_by_cols(df_train,df_test, Configs.HISTORY_RECIPE)

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))  
       
    # with timer("Add scity/bacno latent feature"):
    #     df = pd.read_csv("../features/bacno_latent_features_w_scity.csv")
    #     df_train = df_train.merge(df, on = "bacno", how = "left")
    #     df_test = df_test.merge(df, on = "bacno", how = "left")
    #     df = pd.read_csv("../features/scity_latent_features.csv")
    #     df_train = df_train.merge(df, on = "scity", how = "left")
    #     df_test = df_test.merge(df, on = "scity", how = "left")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer("Add stocn/bacno latent feature"):
    #     df = pd.read_csv("../features/bacno_latent_features_w_stocn.csv")
    #     df_train = df_train.merge(df, on = "bacno", how = "left")
    #     df_test = df_test.merge(df, on = "bacno", how = "left")
    #     df = pd.read_csv("../features/stocn_latent_features.csv")
    #     df_train = df_train.merge(df, on = "stocn", how = "left")
    #     df_test = df_test.merge(df, on = "stocn", how = "left")

    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))

    # with timer('Add time-aggregate features'):
    #     from extraction import merge_and_split_dfs, get_conam_dict_by_day, last_x_day_conam

    #     df, split_df = merge_and_split_dfs(df_train, df_test)
    #     conam_dict = get_conam_dict_by_day(df)

    #     df['last_3_day_mean_conam_per_day'] = last_x_day_conam(3, df, conam_dict)
    #     df['last_7_day_mean_conam_per_day'] = last_x_day_conam(7, df, conam_dict)
    #     df['last_10_day_mean_conam_per_day'] = last_x_day_conam(10, df, conam_dict)
    #     df['last_30_day_mean_conam_per_day'] = last_x_day_conam(30, df, conam_dict)

    #     df_train, df_test = split_df(df)
        
    #     logger.info("Train application df shape: {}".format(df_train.shape))
    #     logger.info("Test application df shape: {}".format(df_test.shape))



    with timer("Run LightGBM with kfold"):
        if args.feature_selection:
            logger.info("==============Feature Selection==============")
            for df in [df_train, df_test]:
                # drop random features (by null hypothesis)
                df.drop(Configs.FEATURE_GRAVEYARD, axis=1, inplace=True, errors='ignore')

                # drop unused features features_with_no_imp_at_least_twice
                df.drop(Configs.FEATURE_USELESSNESS, axis=1, inplace=True, errors='ignore')

                gc.collect()

        for df in [df_train, df_test]:
            df.drop(columns = ["loctm_hour_of_day",
                               "loctm_minute_of_hour", 
                               "loctm_second_of_min",
                               "day_hr_min",
                               "day_hr_min_sec",
                               "cano_locdt_index",
                               "cano_len",
                               "bacno_len",
                               "agg_mchno_mean_conam_in_past_14_days",
                               "agg_mchno_mean_conam_in_past_7_days",
                               ], axis = 1, inplace = True, errors = "ignore")

        logger.info("Train application df shape: {}".format(df_train.shape))
        logger.info("Test application df shape: {}".format(df_test.shape))

    
        ITERATION = (5 if args.TEST_NULL_HYPO else 1)
        feature_importance_df = pd.DataFrame()
        over_iterations_val_auc = np.zeros(ITERATION)
        for i in range(ITERATION):
            logger.info('Iteration %i' %i)
            if args.model == "lgb":    
                iter_feat_imp, over_folds_val_auc = kfold_lightgbm(df_train, df_test, num_folds = args.NUM_FOLDS, args = args, stratified = args.STRATIFIED, seed = args.SEED, logger = logger)
            elif args.model == "xgb":
                iter_feat_imp, over_folds_val_auc = kfold_xgb(df_train, df_test, num_folds = args.NUM_FOLDS, args = args, stratified = args.STRATIFIED, seed = args.SEED, logger = logger)
            else:
                print("Now we only support LightGBM or Xgboost model!")           
            feature_importance_df = pd.concat([feature_importance_df, iter_feat_imp], axis=0)
            over_iterations_val_auc[i] = over_folds_val_auc

        logger.info('============================================\nOver-iterations val f1 score %.6f' %over_iterations_val_auc.mean())
        logger.info('Standard deviation %.6f\n============================================' %over_iterations_val_auc.std())
    
    if args.feature_importance_plot == True:
        from util import display_importances
        display_importances(feature_importance_df, args.model)
        
    feature_importance_df_median = feature_importance_df[["feature", "importance"]].groupby("feature").median().sort_values(by="importance", ascending=False)
    useless_features_df = feature_importance_df_median.loc[feature_importance_df_median['importance'] == 0]
    feature_importance_df_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

    if args.TEST_NULL_HYPO:
        feature_importance_df_mean.to_csv("../result/feature_importance-null_hypo.csv", index = True)
    else:
        feature_importance_df_mean.to_csv("../result/feature_importance.csv", index = True)
        useless_features_list = useless_features_df.index.tolist()
        logger.info('Useless features: \'' + '\', \''.join(useless_features_list) + '\'')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../../dataset/train.csv', type=str)
    parser.add_argument('--test_file', default='../../dataset/test.csv', type=str)
    parser.add_argument('--result_path', default='../result/submission.csv', type=str)
    # lgbm parameters(needed to be filled in with best parameters eventually)
    parser.add_argument('--NUM_FOLDS', default=5, type=int, help='number of folds we split for out-of-fold validation')
    parser.add_argument('--SEED', default=1030, type=int, help='set seed for reproducibility')
    parser.add_argument('--NUM_LEAVES', default=31, type=int, help='Maximum tree leaves for base learners.')
    parser.add_argument('--CPU_USE_RATE', default=1.0, type=float, help='0~1 use how many percentanges of cpu')
    parser.add_argument('--COLSAMPLE_BYTREE', default=1.0, type=float, help = "Subsample ratio of columns when constructing each tree.")
    parser.add_argument('--SUBSAMPLE', default=1.0, type=float, help= " Subsample ratio of the training instance.")
    parser.add_argument('--SUBSAMPLE_FREQ', default=0, type=int, help='Frequence of subsample, <=0 means no enable.')
    parser.add_argument('--MAX_DEPTH', default=-1, type=int, help='Maximum tree depth for base learners, <=0 means no limit.')
    parser.add_argument('--REG_ALPHA', default=0.0, type=float, help = "L1 regularization term on weights.")
    parser.add_argument('--REG_LAMBDA', default=0.0, type=float,  help = "L2 regularization term on weights")
    parser.add_argument('--MIN_SPLIT_GAIN', default=0.0, type=float, help = "Minimum loss reduction required to make a further partition on a leaf node of the tree.")
    parser.add_argument('--MIN_CHILD_WEIGHT', default=0.001, type=float, help= "Minimum sum of instance weight (hessian) needed in a child (leaf).")
    parser.add_argument('--MAX_BIN', default=255, type=int, help='max number of bins that feature values will be bucketed in,  constraints: max_bin > 1')
    parser.add_argument('--SCALE_POS_WEIGHT', default=3.0, type=float, help = "weight of labels with positive class")
    # para
    parser.add_argument('--feature_importance_plot', default=True, type=bool, help='plot feature importance')
    parser.add_argument('--feature_selection', default=False, type=bool, help='drop unused features and random features (by null hypothesis). If true, need to provide features set in list format')
    parser.add_argument('--STRATIFIED', default=True, type=bool, help='use STRATIFIED k-fold. Otherwise, use k-fold')
    parser.add_argument('--TEST_NULL_HYPO', default=False, type=bool, help='get random features by null hypothesis')
    parser.add_argument('--ensemble', default=True, type=bool, help='save testing results with predicted prob for ensemble')
    parser.add_argument('--model', default='lgb', type=str, help='lgb or xgb')
    #
    parser.add_argument('--load_feature', default='False', type=bool, help='determined if use the feature extracted already')
    
    main(parser.parse_args())
