import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector
from sklearn.feature_selection import SequentialFeatureSelector
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

data = pd.read_csv('letter-recognition.data', header=None).to_numpy()
data_hk = data[('H' == data[:, 0]) | ('K' == data[:, 0])]
data_my = data[('M' == data[:, 0]) | ('Y' == data[:, 0])]
data_ab = data[('A' == data[:, 0]) | ('B' == data[:, 0])]


def split_data(samples):
    row, column = samples.shape
    training_rows = int(row * 0.9)
    return samples[0:training_rows, :], samples[training_rows:row, :]


def fit(model, name, dataset, classifier, hyperparameter, x_label, dr_method, dr_require_y=True):
    values = list(hyperparameter.values())[0]
    _min = min(values)
    _max = max(values)
    interval = (_max - _min) / (len(values) - 1)
    _range = np.arange(_min - interval, _max + interval, interval)

    def plot(data_type):
        plt.xticks(_range)
        plt.scatter(list(hyperparameter.values())[0], clf.cv_results_['mean_test_score'])
        plt.title(f"{model}_{data_type}_{name}")
        plt.xlabel(x_label)
        plt.ylabel("performance")
        plt.savefig(f"figures/{model}/{name}_{data_type}")
        plt.clf()

    training, testing = split_data(dataset)
    # x = StandardScaler().fit_transform(training[:, 1:])
    x = training[:, 1:]
    y = training[:, 0]
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV.score
    clf = GridSearchCV(
        estimator=classifier,
        param_grid=hyperparameter,
        cv=5
    )
    clf.fit(x, y)
    plot('normal')
    # print(clf.cv_results_)
    # print(clf.cv_results_['params'])
    # plt.plot(raw[:, column], predicted, color="yellow")
    # dr = dimension reduction
    if dr_require_y:
        dr_x = dr_method.fit_transform(x, y)
    else:
        dr_x = dr_method.fit_transform(x)
    clf.fit(dr_x, y)
    plot('dimension_reduction')
    # print(clf.best_estimator_)


def train_models(dataset, name):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    fit('knn', name, dataset, KNeighborsClassifier(), {'n_neighbors': [1, 2, 3, 4, 5]}, 'n_neighbors',
        PCA(n_components=4), False)
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    fit('decision_tree', name, dataset, DecisionTreeClassifier(), {'max_depth': [1, 2, 3, 4, 5]}, 'max_depth',
        SelectKBest(k=4))
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    fit('random_forest', name, dataset, RandomForestClassifier(), {'max_depth': [1, 2, 3, 4, 5]}, 'max_depth',
        SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), n_features_to_select=4, direction='backward'))
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    fit('svm', name, dataset, SVC(), {'C': [1, 2, 3, 4, 5]}, 'C',
        SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), n_features_to_select=4))
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    fit('ann', name, dataset, MLPClassifier(max_iter=1500), {'hidden_layer_sizes': [100, 150, 200, 250, 300]},
        'hidden_layer_sizes', SelectKBest(chi2, k=4))


train_models(data_hk, 'HK')
train_models(data_my, 'MY')
train_models(data_ab, 'AB')
