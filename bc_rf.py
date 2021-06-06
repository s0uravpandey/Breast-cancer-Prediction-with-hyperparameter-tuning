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

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy',max_depth = 10, min_samples_leaf=1,min_samples_split=5,bootstrap = False, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))
#grid search
from sklearn.model_selection import GridSearchCV
parameters = [{  'bootstrap': [True, False],
                 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                 'min_samples_leaf': [1,2,3,4],
                 'min_samples_split': [5,10],
                 'n_estimators': [200, 400, 600, 800, 1000]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
