# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:18:41 2020

@author: namthatman
"""

# Titanic: Machine Learning from Disaster

# Inspired the wonderful cleaning techniques from the notebook by Manav Sehgal: https://www.kaggle.com/startupsci/titanic-data-science-solutions?fbclid=IwAR1EyFMhML2Iy42wCuj-Wld4O6lPNkcThqgjIZcxHVPKr5kUe0PDuDQ-MpY

# How to get the Dataset: https://www.kaggle.com/c/titanic/data


# Importing the libraries
import numpy as np
import pandas as pd


# Importing Machine Learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score


# Importing the dataset
X_df = pd.read_csv('train.csv')
Y_df = pd.read_csv('test.csv')


# Simple exploration
#X_df.head()
#X_df.tail()
#X_df.info()
#X_df.describe()
#X_df.describe(include=['O'])


# Analyzing and Visualizing some features using Tableau for decision making


# Dropping Ticker, Cabin features
drop_list = ['Ticket', 'Cabin']
X_df = X_df.drop(drop_list, axis = 1)
Y_df = Y_df.drop(drop_list, axis = 1)


# Extracting Name feature    
X_df['Title'] = pd.Series(X_df['Name']).str.extract(' ([A-Za-z]+)\.', expand=False)
Y_df['Title'] = pd.Series(Y_df['Name']).str.extract(' ([A-Za-z]+)\.', expand=False)
#pd.crosstab(X_df['Title'], X_df['Sex'])
X_df['Title'] = X_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
X_df['Title'] = X_df['Title'].replace('Mlle', 'Miss')
X_df['Title'] = X_df['Title'].replace('Ms', 'Miss')
X_df['Title'] = X_df['Title'].replace('Mme', 'Mrs')

Y_df['Title'] = Y_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
Y_df['Title'] = Y_df['Title'].replace('Mlle', 'Miss')
Y_df['Title'] = Y_df['Title'].replace('Ms', 'Miss')
Y_df['Title'] = Y_df['Title'].replace('Mme', 'Mrs')

# Encoding Title feature
labelencoder_X = LabelEncoder()
X_df['Title'] = labelencoder_X.fit_transform(X_df['Title'])
labelencoder_Y = LabelEncoder()
Y_df['Title'] = labelencoder_Y.fit_transform(Y_df['Title'])

# Dropping Name feature
X_df = X_df.drop(['Name', 'PassengerId'], axis = 1)
Y_df = Y_df.drop(['Name'], axis = 1)


# Converting Sex feature
X_df['Sex'] = X_df['Sex'].map({'male': 0, 'female': 1})
Y_df['Sex'] = Y_df['Sex'].map({'male': 0, 'female': 1})


# Filling missing value for Age by other correlated features (Age-Pclass-Gender)
guess_ages = np.zeros((2,3))
for dataset in [X_df, Y_df]:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# Grouping Age feature into bands and encoding 
for dataset in [X_df, Y_df]:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    

# Creating Family feature from SibSp and Parch
for dataset in [X_df, Y_df]:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch']
    
# Dropping SibSp and Parch
X_df = X_df.drop(['SibSp', 'Parch'], axis = 1)
Y_df = Y_df.drop(['SibSp', 'Parch'], axis = 1)


# Filling missing value for Embarked feature
most_embarked = X_df.Embarked.dropna().mode()[0]
for dataset in [X_df, Y_df]:
    dataset['Embarked'] = dataset['Embarked'].fillna(most_embarked)
    
# Encoding Embarked feature
for dataset in [X_df, Y_df]:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    

# Filling missing value for Fare feature
X_df['Fare'].fillna(X_df['Fare'].dropna().median(), inplace=True)
Y_df['Fare'].fillna(Y_df['Fare'].dropna().median(), inplace=True)

# Grouping Fare feature into bands and Encoding
percentiles = [np.percentile(X_df['Fare'], 25), np.percentile(X_df['Fare'], 50), np.percentile(X_df['Fare'], 75)]
for dataset in [X_df, Y_df]:
    dataset.loc[ dataset['Fare'] <= percentiles[0], 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > percentiles[0]) & (dataset['Fare'] <= percentiles[1]), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > percentiles[1]) & (dataset['Fare'] <= percentiles[2]), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > percentiles[2], 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# Splitting the dataset into the Training set and Test set
X_train = X_df.drop(['Survived'], axis = 1)
Y_train = X_df['Survived']
X_test = Y_df.drop("PassengerId", axis = 1).copy()
#Y_test


# Feature Scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Logistic Regression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train, Y_train)
pred_lr = lr.predict(X_test)
accuracy_lr = cross_val_score(estimator = lr, X = X_train, y = Y_train, cv = 10)
accuracy_lr = accuracy_lr.mean()


# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, Y_train)
pred_knn = knn.predict(X_test)
accuracy_knn = cross_val_score(estimator = knn, X = X_train, y = Y_train, cv = 10)
accuracy_knn = accuracy_knn.mean()


# Support Vector Machines
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(X_train, Y_train)
pred_svm = svm.predict(X_test)
accuracy_svm = cross_val_score(estimator = svm, X = X_train, y = Y_train, cv = 10)
accuracy_svm = accuracy_svm.mean()


# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
pred_nb = nb.predict(X_test)
accuracy_nb = cross_val_score(estimator = nb, X = X_train, y = Y_train, cv = 10)
accuracy_nb = accuracy_nb.mean()


# Decision Tree
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train, Y_train)
pred_dt = dt.predict(X_test)
accuracy_dt = cross_val_score(estimator = dt, X = X_train, y = Y_train, cv = 10)
accuracy_dt = accuracy_dt.mean()


# Random Forest
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)
accuracy_rf = cross_val_score(estimator = rf, X = X_train, y = Y_train, cv = 10)
accuracy_rf = accuracy_rf.mean()


# Evaluating models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Decision Tree'],
    'Score': [accuracy_svm, accuracy_knn, accuracy_lr, 
              accuracy_rf, accuracy_nb, accuracy_dt]})
models.sort_values(by='Score', ascending=False)
