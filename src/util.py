import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import gc
from tqdm import tqdm

def string_padding(s):
    """
    Test case:

    s = "819"
    after padding
    s = "000819"
    """
    while len(s)!=6:
        s = "0" + s
    return s

def s_to_time_format(s):
    """
    Test case:
    
    s = "153652"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "15:36:52", "It should be the same"
    s = "91819"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "09:18:19", "It should be the same"
    s = "819"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "00:08:19", "It should be the same"
    s = "5833"
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    assert datetime_str == "00:58:33", "It should be the same"
 

    """
    s = string_padding(s)
    S, M, H = s[-2:], s[-4:-2], s[:-4]
    datetime_str = "{}:{}:{}".format(H,M,S)
    return datetime_str

def string_to_datetime(datetime_str):
    """
    input: '09:18:19'
    after the function
    return datetime.datetime(1900, 1, 1, 9, 18, 19)
    Please ignore 1900(Year),1(month), 1(day)
    """
    from datetime import datetime
    datetime_object = datetime.strptime(datetime_str, '%H:%M:%S')
    return datetime_object

def hour_to_range(hr):
    if hr > 22 and hr <= 3:
        return 'midnight'
    elif hr > 3 and hr <= 7:
        return 'early_morning'
    elif hr > 7 and hr <= 11:
        return 'morning'
    elif hr > 11 and hr <= 14:
        return 'noon'
    elif hr >14 and hr <= 17:
        return 'afternoon'
    else:
        return 'night'

def num_transaction(df,target = "cano"):
    tmp_cano_size = df.groupby(target).size().to_frame("{}_len".format(target)).reset_index()
    df = df.merge(tmp_cano_size, on = target, how = "left")
    return df

def if_conam_zero(df):
    df["if_conam_zero"] = [1 if int(i)==0 else 0 for i in df.conam]
    return df

def time_elapsed_between_last_transactions(df):
    if len(df) > 1:
        df.time_elapsed_between_last_transactions = [df.time_elapsed_between_last_transactions.iloc[0] for i in range(len(df))]
        return df.time_elapsed_between_last_transactions
    else:
        return df.time_elapsed_between_last_transactions

def _time_elapsed_between_last_transactions(df):
    #return df.locdt.diff(periods=1).fillna(0)
    return df.locdt.diff(periods=1)

def num_transaction_in_past_n_days(df, n):
    """
    Calculate how many transaction that this user have in the past n days
    """
    current_day_at_this_transaction = df.locdt.tolist()
    output = []
    for current_date in current_day_at_this_transaction:
        history_date = current_date-n
        c = 0
        for i in current_day_at_this_transaction:
            if (i >= history_date) & (i < current_date):
                c+=1
        output.append(c)
    return pd.Series(output) # return Series instead of list

def rolling_stats_target_by_cols(df_train, df_test, recipe, window = 2):
    df = pd.concat([df_train, df_test], axis = 0)
    #df.sort_values(by = ["bacno","locdt"], inplace = True)
    
    for m in range(len(recipe)):
        cols = recipe[m][0]
        #print (cols + ["locdt"])
        df.sort_values(by = cols + ["locdt"], inplace = True)
        for n in range(len(recipe[m][1])):
            target = recipe[m][1][n][0]
            method = recipe[m][1][n][1]
            name_grouped_target = method+"_"+target+'_BY_'+'_'.join(cols)+"_"+"in_past_{}_transactions".format(window)
            #print (name_grouped_target)
            if method == "mean":
                df[name_grouped_target] = df.groupby(cols)[target].rolling(window=window).mean().values
                
    df_train = df[~df.fraud_ind.isnull()]
    df_test = df[df.fraud_ind.isnull()]

    df_test.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
    del df
    gc.collect()
    
    return df_train, df_test

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    out = []
    for i in range(0, len(l), n):
        out.append(l[i:i + n])
    return out

def group_target_by_cols_split_by_users(df_train, df_test, recipe, user_column = "bacno", num_partitions = 10):
    """
    Split and merge dataframe for memory problem
    """
    df = pd.concat([df_train, df_test], axis = 0)

    df_users = df[user_column].unique().tolist()
    train_users = df_train[user_column].unique().tolist()
    test_users = df_test[user_column].unique().tolist()      
    #o_feats = df.columns.tolist()
    
    output = []
    for partition_users in tqdm(chunks(df_users,num_partitions)):
        print ("partition_users",len(partition_users))
        tmp_df_ls = [] # including partiton of the users w new features we just created
        c = 0
        for m in range(len(recipe)):
            cols = recipe[m][0]
            for n in range(len(recipe[m][1])):
                target = recipe[m][1][n][0]
                method = recipe[m][1][n][1]
                name_grouped_target = method+"_"+target+'_BY_'+'_'.join(cols)
                df_split_by_users = df[df[user_column].isin(partition_users)]
                tmp = df_split_by_users[cols + [target]].groupby(cols).agg(method)
                tmp = tmp.reset_index().rename(index=str, columns={target: name_grouped_target})
                if c!= 0:
                    tmp_df_ls.append(df_split_by_users.merge(tmp, how='left', on=cols)[name_grouped_target])
                else:
                    tmp_df_ls.append(df_split_by_users.merge(tmp, how='left', on=cols))
                c+=1
                # reduce memory
                del df_split_by_users,tmp
                gc.collect()             
        tmp_df_ls = pd.concat(tmp_df_ls, axis = 1)
        output.append(tmp_df_ls)
        # reduce memory
        del tmp_df_ls
        gc.collect()             

    df = pd.concat(output, axis = 0) # including all the users w new features we just created

    df_train = df[df[user_column].isin(train_users)]
    df_test = df[df[user_column].isin(test_users)]
    df_test.drop(columns = ["fraud_ind"], axis = 1, inplace = True)
    del df
    gc.collect()  
    return df_train, df_test

# Display/plot feature importance
def display_importances(feature_importance_df_, model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if model == "lgb":
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(32, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout
        plt.savefig('../result/lgbm_importances.png')
    elif model == "xgb":
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(32, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('Xgboost Features (avg over folds)')
        plt.tight_layout
        plt.savefig('../result/xgb_importances.png')
    else:
        print("Now we only support LightGBM or Xgboost model!")  

def add_auto_encoder_feature(df_raw, df, autoencoder, add_reconstructed_vec = True):

    predictions = autoencoder.predict(df) # get reconstructed vector, 2-D, [num_samples, num_features]
    mse = np.mean(np.power(df - predictions, 2), axis=1) # get reconstructed error, 1-D, [num_samples,]

    if add_reconstructed_vec == True:
        df = pd.DataFrame(predictions, columns=["reconstructed_dim_{}".format(i) for i in range(predictions.shape[1])])
        df["reconstruction_error"] = mse
    else:
        df = pd.DataFrame({"reconstruction_error": mse})
    out = pd.concat([df_raw.reset_index(drop = True), df.reset_index(drop = True)], axis = 1)

    assert len(out)==len(df_raw)==len(df), "it should be same"

    return out

def lgb_f1_score(y_true, y_pred):
    """evaluation metric"""
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True

# lgb model
def kfold_lightgbm(df_train, df_test, num_folds, args, logger, stratified = False, seed = int(time.time())):
    """
    LightGBM GBDT with KFold or Stratified KFold
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import multiprocessing   
    import lightgbm as lgb
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    #train_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in ["fraud_ind"]]
    # k-fold
    if args.TEST_NULL_HYPO:
        # shuffling our label for feature selection
        df_train['fraud_ind'] = df_train['fraud_ind'].copy().sample(frac=1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['fraud_ind'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['fraud_ind'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['fraud_ind'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        if args.TEST_NULL_HYPO:
            clf = lgb.LGBMClassifier(
                n_jobs = 3,
                # nthread=int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
                n_estimators=10000,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )
        else:
            clf = lgb.LGBMClassifier(
                n_jobs = 19,
                n_estimators=20000,
                learning_rate=0.001, # 0.02
                num_leaves=int(args.NUM_LEAVES),
                colsample_bytree=args.COLSAMPLE_BYTREE,
                subsample=args.SUBSAMPLE,
                subsample_freq=args.SUBSAMPLE_FREQ,
                max_depth=args.MAX_DEPTH,
                reg_alpha=args.REG_ALPHA,
                reg_lambda=args.REG_LAMBDA,
                min_split_gain=args.MIN_SPLIT_GAIN,
                min_child_weight=args.MIN_CHILD_WEIGHT,
                max_bin=args.MAX_BIN,
                silent=-1,
                verbose=-1,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )

        if args.TEST_NULL_HYPO:
            clf.fit(train_x, 
                    train_y, 
                    eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= "auc", 
                    verbose= True, 
                    early_stopping_rounds= 100, 
                    categorical_feature='auto') # early_stopping_rounds= 200
        else:
            clf.fit(train_x, 
                    train_y, 
                    eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric= lgb_f1_score, 
                    verbose= False, 
                    early_stopping_rounds= 200, 
                    categorical_feature='auto') # early_stopping_rounds= 200
        # probabilty belong to class1(fraud)
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logger.info('Fold %2d val f1-score : %.6f' % (n_fold + 1, lgb_f1_score(valid_y, oof_preds[valid_idx])[1]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    #print('---------------------------------------\nOver-folds train f1-score %.6f' % lgb_f1_score(df_train['fraud_ind'], train_preds)[1])
    logger.info('---------------------------------------\n')
    over_folds_val_score = lgb_f1_score(df_train['fraud_ind'], oof_preds)[1]
    logger.info('Over-folds val f1-score %.6f\n---------------------------------------' % over_folds_val_score)
    # Write submission file and plot feature importance

    if args.ensemble:
        df_test.loc[:,'fraud_ind'] = sub_preds
        df_test[['txkey', 'fraud_ind']].to_csv("../result/lgb.csv", index= False)

    df_test.loc[:,'fraud_ind'] = np.round(sub_preds)
    df_test[['txkey', 'fraud_ind']].to_csv(args.result_path, index= False)
    
    return feature_importance_df, over_folds_val_score

def xgb_f1_score(y_pred, y_true):
    """evaluation metric"""
    y_hat = np.round(y_pred)
    y_true = y_true.get_label()
    #print ('f1-score', f1_score(y_true, y_hat))
    return 'f1-score-error', 1-f1_score(y_true, y_hat) # error

# xgb model
def kfold_xgb(df_train, df_test, num_folds, args, logger, stratified = False, seed = int(time.time())):
    """
    Xgboost with KFold or Stratified KFold
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import multiprocessing   
    from xgboost import XGBClassifier
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    #train_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in df_train.columns if f not in ["fraud_ind"]]
    # k-fold
    if args.TEST_NULL_HYPO:
        # shuffling our label for feature selection
        df_train['fraud_ind'] = df_train['fraud_ind'].copy().sample(frac=1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['fraud_ind'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['fraud_ind'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['fraud_ind'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        if args.TEST_NULL_HYPO:
            clf = lgb.LGBMClassifier(
                nthread=int(multiprocessing.cpu_count()*args.CPU_USE_RATE),
                n_estimators=10000,
                learning_rate=0.02,
                objective='binary:logistic',
                booster='gbtree',
                num_leaves=127,
                max_depth=args.MAX_DEPTH,
                random_state=seed,
                )
        else:
            clf = XGBClassifier(
                n_jobs = 1,
                max_depth=3,
                learning_rate=0.05,
                n_estimators=10000,
                silent=True,
                objective='binary:logistic',
                booster='gbtree',
                gamma=0, 
                min_child_weight=1, 
                max_delta_step=0, 
                subsample=0.8, 
                colsample_bytree=1, 
                colsample_bylevel=1, 
                colsample_bynode=0.8, 
                reg_alpha=0, 
                reg_lambda=1e-05,
                random_state=seed,
                scale_pos_weight=args.SCALE_POS_WEIGHT
                )
        clf.fit(train_x, 
                train_y, 
                eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= xgb_f1_score, 
                verbose= False, 
                early_stopping_rounds= 100, 
                #categorical_feature='auto'
                ) # early_stopping_rounds= 200
        # probabilty belong to class1(fraud)
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds += clf.predict_proba(df_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logger.info('Fold %2d val f1-score : %.6f' % (n_fold + 1, lgb_f1_score(valid_y, oof_preds[valid_idx])[1]))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    #print('---------------------------------------\nOver-folds train f1-score %.6f' % lgb_f1_score(df_train['fraud_ind'], train_preds)[1])
    logger.info('---------------------------------------\n')
    over_folds_val_score = lgb_f1_score(df_train['fraud_ind'], oof_preds)[1]
    logger.info('Over-folds val f1-score %.6f\n---------------------------------------' % over_folds_val_score)

    # Write submission file and plot feature importance
    if args.ensemble:
        df_test.loc[:,'fraud_ind'] = sub_preds
        df_test[['txkey', 'fraud_ind']].to_csv("../result/xgb.csv", index= False)

    df_test.loc[:,'fraud_ind'] = np.round(sub_preds)
    df_test[['txkey', 'fraud_ind']].to_csv(args.result_path, index= False)
    
    return feature_importance_df, over_folds_val_score
    