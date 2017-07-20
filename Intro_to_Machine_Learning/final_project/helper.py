import sys
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit
from sklearn import preprocessing


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from pprint import pprint



def gfr(data_dict, features_list):

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    clf = DecisionTreeClassifier()
    clf.fit(features, labels)

    # What is the importance of the most important feature?
    importances = clf.feature_importances_
    print '\nFeature importance in descending order:'
    print [X for Y, X in sorted(zip(importances, features_list[1:]), reverse=True)]
    print sorted(importances, reverse=True)
    
    # display feature importance
    f, ax = plt.subplots(figsize=(10, 7))
    ax.bar( range(len(clf.feature_importances_)), clf.feature_importances_)
    ax.set_title("Initial Feature Importances")
    f.show()
    


def calculate_fraction(poi_messages, all_messages):
    """
    return the fraction of messages to/from that person that are from/to a POI
    """

    if poi_messages == 'NaN' or all_messages == 'NaN' or poi_messages == 0 or all_messages == 0:
        fraction = 0.
    else:
        fraction = float(poi_messages) / float(all_messages)

    return fraction


def select_k_best(data_dict, features_list, k):

    """ runs scikit-learn's SelectKBest feature selection to get best featuers
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_

    unsorted_results = zip(features_list[1:], scores)
    sorted_results = list(reversed(sorted(unsorted_results, key=lambda x: x[1])))

    # print sorted_results
    k_best_features = dict(sorted_results[:k])
    print "{0} Best Features: {1}\n".format(k, k_best_features.keys())    
    return k_best_features


def plot_roc(clf, features_test, labels_test):
    """
    - Insensitive to data sets with unbalanced class proportions unlike accuracy
    - Shows classifiers performance for all values of the discrimination threshold unlike precision & recall
    - Plots the classifiers recall against its fall-out/false positive rate
    - Fall out rate = FP/(TN+FP)
    - AUC is the area under the ROC curve. It reduces the ROC curve to a single value which represents
      the expected performance of the classifier
    - Dashed line represents classifier which predicts classes randomly(auc=0.5)
    - Solid curve is for classifier which outperforms/underperforms random guessing
    """
    predictions = clf.predict_proba(features_test)
    false_positive_rate, recall, thresholds = roc_curve(labels_test, predictions[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()
    
