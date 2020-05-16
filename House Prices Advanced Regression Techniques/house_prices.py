# -*- coding: utf-8 -*-
'''
Created on Tue May  5 13:55:36 2020

@author: namthatman
'''

# House Prices: Advanced Regression Techniques

# Inspired the wonderful cleaning techniques from the notebook by Serigne: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard?fbclid=IwAR1XYPFNe5UGc8YjGaLE4tUzBC_YccObll6HV9d9DToGgmRkkdE6AM4nuaY

# How to get the Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data


# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew


# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# More Analyzing and Visualizing some features using Tableau for decision making


# Dropping Id column
train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)


# Detecting Outliers
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# Drop Outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# Transforming target into normally distributed form using log-transformation
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'] , fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# Missing Values
n_train = train.shape[0]
n_test = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


# Missing Ratio
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# Correlation Map
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# Handling Missing Values

# PoolQC, NA means No Pool
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

# MiscFeature, NA means No Misc Feature
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

# Alley, NA means No Alley
all_data['Alley'] = all_data['Alley'].fillna('None')

# Fence, NA means No Fence
all_data['Fence'] = all_data['Fence'].fillna('None')

# FireplaceQu, NA means No Fireplace
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

# LotFrontage, fill in NA by the median LotFrontage of the neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# GarageType, GarageFinish, GarageQual and GarageCond, NA means No Garage, ... (categorical)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
# GarageYrBlt, GarageArea and GarageCars, NA means No Car (No Garage = No Car) (numerical)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath, NA means No Basement (numerical)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2, NA means No Basement (categorical)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
# MasVnrArea and MasVnrType, NA means No masonry veneer
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

# MSZoning, fill in NA with 'RL', the most common value
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Utilities, Since all records are 'AllPub', except for one 'NoSeWa' and 2 NA, we can then safely remove it.
all_data = all_data.drop(['Utilities'], axis=1)

# Functional, NA means typical (data description)
all_data['Functional'] = all_data['Functional'].fillna('Typ')

# Electrical, fill in NA with 'SBrkr'
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenQual, fill in NA with 'TA', the most common value
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior1st and Exterior2nd, only one NA, fill in NA with the most common value
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType, fill in NA with 'WD', the most common value
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# MSSubClass, NA means No building class
all_data['MSSubClass'] = all_data['MSSubClass'].fillna('None')

# Checking Remaining Missing Values
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# Feature Engineering

# MSSubClass = The building class (numerical -> categorical)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# Label Encoding categorical variables that contain (ordering, ranking, quality condition, etc.)
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    le = LabelEncoder() 
    le.fit(list(all_data[c].values)) 
    all_data[c] = le.transform(list(all_data[c].values))
    

# Adding one feature TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# Handling Skewed Features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

# BoxCox Transformation on Skewed Features
skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], 0.15)
    
# BoxCox setting lambda = 0  is equivalent to log1p
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
    

# One-Hot Encoding (Dummies) categorical features
all_data = pd.get_dummies(all_data)


# Getting new train and test sets
train = all_data[:n_train]
test = all_data[n_train:]


# Modelling

# Importing Machine Learning libraries
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# Models

Lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

SVMR = make_pipeline(RobustScaler(), SVR(kernel='poly', degree=2, coef0=2.5))

KRR = KernelRidge(alpha=0.6, kernel='poly', degree=2, coef0=2.5)

RForest = RandomForestRegressor(n_estimators=3000, max_depth=4, min_samples_leaf=15, 
                                min_samples_split=10, max_features='sqrt', random_state =5 )

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

Xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

Lgbm = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# Cross Validation
def rmse_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=7).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv = kf))
    return(rmse)

# Scores
score = rmse_cv(Lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(ENet)
print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(SVMR)
print("\nSupport Vector Regressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(KRR)
print("\nKernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(RForest)
print("\nRandom Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(GBoost)
print("\nGradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(Xgboost)
print("\nXGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(Lgbm)
print("\nLightGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Averaging Models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)

        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 

# Averaging Models Score
AVM = AveragingModels(models = (Lasso, ENet, SVMR, KRR, GBoost))
score = rmse_cv(AVM)
print("\nAveraged models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Final Training and Prediction
AVM.fit(train, y_train)
AVM_train_pred = AVM.predict(train)
AVM_pred = np.expm1(AVM.predict(test))
print(np.sqrt(mean_squared_error(y_train, AVM_train_pred)))

Xgboost.fit(train, y_train)
Xgboost_train_pred = Xgboost.predict(train)
Xgboost_pred = np.expm1(Xgboost.predict(test))
print(np.sqrt(mean_squared_error(y_train, Xgboost_train_pred)))

Lgbm.fit(train, y_train)
Lgbm_train_pred = Lgbm.predict(train)
Lgbm_pred = np.expm1(Lgbm.predict(test))
print(np.sqrt(mean_squared_error(y_train, Lgbm_train_pred)))

print('RMSE score on train data:')
print(np.sqrt(mean_squared_error(y_train, (AVM_train_pred + Xgboost_train_pred + Lgbm_train_pred)/3 )))

# Ensemble Prediction
prediction = (AVM_pred + Xgboost_pred + Lgbm_pred)/3
