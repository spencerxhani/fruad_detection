### How to run

0. Please create a folder named dataset in current path and put train.csv and test.csv into this folder

1. Run the below code under src directory and in order
```
make mf
```
```
make mf-mchno
```
```
make train-lgb
```
### Reuslt
By defualt, it will generate the below file in a folder called result

* submission.csv
* log
* feature_importance.csv
* lgbm_importances.png
