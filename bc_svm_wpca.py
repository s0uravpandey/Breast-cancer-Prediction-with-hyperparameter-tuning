#malignant cases = 1   benign cases = 0

#Breast Cancer
# Library importing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('data_f.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Data visualizing
sns.pairplot(dataset, hue = 'diagnosis', vars = ['radius_1ean','texture_1ean','peri1eter_1ean','area_1ean'])
sns.countplot(dataset['diagnosis'])
%matplotlib qt

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',C=10,gamma = 0.01, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]},
              {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1,0.01,0.001,0.0001]},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly'],'degree': [2,3,4,5,6],'gamma': [0.1,0.01,0.001,0.0001]}] 
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

