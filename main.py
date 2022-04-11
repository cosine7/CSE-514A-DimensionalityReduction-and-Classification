import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel

data = pd.read_csv('letter-recognition.data', header=None).to_numpy()
data_hk = data[('H' == data[:, 0]) | ('K' == data[:, 0])]
data_my = data[('M' == data[:, 0]) | ('Y' == data[:, 0])]
data_ab = data[('A' == data[:, 0]) | ('B' == data[:, 0])]


def split_data(samples):
    row, column = samples.shape
    training_rows = int(row * 0.9)
    return samples[0:training_rows, :], samples[training_rows:row, :]


def train_model(dataset, estimator, param_grid):
    training, testing = split_data(dataset)
    x = StandardScaler().fit_transform(training[:, 1:])
    y = training[:, 0]
    clf = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5
    )
    clf.fit(x, y)
    # print(clf.cv_results_['params'])
    print(clf.best_estimator_)


# train_model(data_hk, KNeighborsClassifier(), {'n_neighbors': [1, 2, 3, 4, 5]})
# # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# train_model(data_hk, DecisionTreeClassifier(), {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
# # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# train_model(data_hk, RandomForestClassifier(), {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
# # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# train_model(data_hk, SVC(), {'kernel': ('linear', 'poly', 'rbf', 'sigmoid')})
# # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
# train_model(data_hk, MLPClassifier(max_iter=1000), {'learning_rate': ['constant', 'invscaling', 'adaptive']})


# def reduce_dimension(dataset):
training, testing = split_data(data_hk)
X = training[:, 1:]
y = training[:, 0]
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = PCA(n_components=4)
# pca.fit(X)
print(pca.fit_transform(X))
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector
print(SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), n_features_to_select=4).fit_transform(X, y))

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
print(SelectFromModel(estimator=DecisionTreeClassifier()).fit_transform(X, y))
