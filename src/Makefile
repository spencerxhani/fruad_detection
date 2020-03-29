null-importance:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_null.csv --feature_selection True --TEST_NULL_HYPO True
train-lgb:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_lgb.csv 
train-lgb-fs:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_lgb.csv --feature_selection True
train-lgb-fs-lf:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_lgb.csv --feature_selection True --load_feature True
train-lgb-fs-best:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_lgb.csv \
		--feature_selection True --SUBSAMPLE 0.8890783754749252 --COLSAMPLE_BYTREE 0.9350060741234096 --MIN_CHILD_WEIGHT  39.245929638036316 \
		--MIN_SPLIT_GAIN 0.011827442586893323 --NUM_LEAVES 37 --SCALE_POS_WEIGHT 2.2901795866814174 --REG_ALPHA 4.89309171116382 \
		--REG_LAMBDA 2.397475692650171 --MAX_BIN 301
train-xgb:
		python main.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --result_path ../result/submission_xgb.csv --model xgb 
mf:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features
mf-mchno:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name mchno
mf-contp:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name contp
mf-scity:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name scity
mf-stocn:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --column_name stocn
mf-cano-mchno:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name cano --column_name mchno
mf-mchno-loctm_hour_of_day:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name mchno --column_name loctm_hour_of_day
mf-cano-locdt:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name cano --column_name locdt
mf-mchno-locdt:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name mchno --column_name locdt
mf-bacno-locdt:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name bacno --column_name locdt
mf-stocn-locdt:
		python matrix_factorization.py --train_file ../../dataset/train.csv --test_file ../../dataset/test.csv --latent_feature_path ../features --row_name stocn --column_name locdt
