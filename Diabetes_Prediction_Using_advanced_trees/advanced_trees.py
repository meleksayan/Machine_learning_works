# ############## Advanced Trees ######################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost #

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("Machine_learning_works/datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# ##### Random Forests ######################
rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
cv_results = cross_validate(rf_model, X, y,
                            cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  # 0.75
cv_results['test_f1'].mean()    # 0.61
cv_results['test_roc_auc'].mean()  # 0.82

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  # 0.76
cv_results['test_f1'].mean()        # 0.64
cv_results['test_roc_auc'].mean()   # 0.82

# ################### GBM #######################

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  # 0.75
cv_results['test_f1'].mean()        # 0.63
cv_results['test_roc_auc'].mean()   # 0.82

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  # 0.77
cv_results['test_f1'].mean()        # 0.66
cv_results['test_roc_auc'].mean()   # 0.83

# ################## XGBoost ######################
xg_model = XGBClassifier(random_state=17)
xg_model.get_params()
cv_results = cross_validate(xg_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  # 0.74
cv_results['test_f1'].mean()        # 0.62
cv_results['test_roc_auc'].mean()   # 0.79

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}
xgboost_best_grid = GridSearchCV(xg_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_final = xg_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()  # 0.76
cv_results['test_f1'].mean()        # 0.64
cv_results['test_roc_auc'].mean()   # 0.81

# #################### LightGBM ###########################
lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.74
cv_results['test_f1'].mean()        # 0.62
cv_results['test_roc_auc'].mean()   # 0.80



lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.76
cv_results['test_f1'].mean()        # 0.63
cv_results['test_roc_auc'].mean()   # 0.81

# Hyperparameter optimization
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.76
cv_results['test_f1'].mean()        # 0.61
cv_results['test_roc_auc'].mean()   # 0.82


# Hiperparametre optimization only for  n_estimators.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.76
cv_results['test_f1'].mean()        # 0.61
cv_results['test_roc_auc'].mean()   # 0.82


# ##################CatBoost #############################
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.77
cv_results['test_f1'].mean()        # 0.65
cv_results['test_roc_auc'].mean()   # 0.83


catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.77
cv_results['test_f1'].mean()        # 0.63
cv_results['test_roc_auc'].mean()   # 0.84


# ############Hyperparameter Optimization with RandomSearchCV (BONUS)###################
rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  #0.76
cv_results['test_f1'].mean()        #0.61
cv_results['test_roc_auc'].mean()   #0.83











