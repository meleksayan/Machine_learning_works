#############################################
# Diabetes Prediction with Logistic Regression
#############################################

# Aim: Develop a machine learning model that can predict whether a person has diabetes based on their features?"

# it contains data from a diabetes study conducted on Pima Indian women aged 21 and over,
# living in Phoenix, the 5th largest city in the state of Arizona, USA

# It consists of 768 observations and 8 numerical independent variables. The target variable is labeled as "Outcome",
# where 1 indicates a positive diabetes test result and 0 indicates a negative result.

# Features
# Pregnancies: number of the pregnancies
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

# ########### starting #####################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("Machine_learning_works/datasets/diabetes.csv")

##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()
sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts()/len(df)

##########################
# Feature'ların Analizi
##########################

df.head()
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]


# for col in cols:
#     plot_numerical_col(df, col)
df.describe().T

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)


# ############ Data Preprocessing #########################

df.shape
df.head()

df.isnull().sum()

df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])
    # RobustScaler, veri setindeki değerleri medyan ve IQR (Interquartile Range) kullanarak ölçeklendirir
    # fit_transform():
    # fit(): medyan ve IQR değerlerini hesaplar.
    # transform(): sütundaki her değeri yukarıdaki formüle göre dönüştürür.

df.head()

# ############### Model and Prediction ###################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)     # axis = 1 demek sutunları hedef al demektir.

log_model = LogisticRegression().fit(X, y)
print("Intercept:", log_model.intercept_)   # intercept → başlangıç log-odds
print("Coefficients:", log_model.coef_)     # coef → her özelliğin Outcome’a etkisi

y_pred = log_model.predict(X)
y_pred[0:10]   # Modelin ilk 10 örnek için tahmini Outcome değerleri.
y[0:10]        # Gerçek Outcome değerleri.


