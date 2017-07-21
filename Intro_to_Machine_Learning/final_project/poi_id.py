# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from pprint import pprint

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from helper import *

from matplotlib import pyplot

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')

from sklearn.cross_validation import train_test_split, cross_val_score
import evaluate



###########################################
######## Step 1: Select Features ##########
###########################################

# 3 types of features have been provided
financial_features = [
                      'salary', 
                      'deferral_payments', 
                      'total_payments', 
                      'loan_advances', 
                      'bonus', 
                      'restricted_stock_deferred', 
                      'deferred_income', 
                      'total_stock_value', 
                      'expenses', 
                      'exercised_stock_options', 
                      'other', 
                      'long_term_incentive', 
                      'restricted_stock', 
                      'director_fees'
                    ]

# (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
email_features = [
                  'to_messages',  
                  'from_poi_to_this_person', 
                  'from_messages', 
                  'from_this_person_to_poi', 
                  'shared_receipt_with_poi'
                 ] 

poi = ['poi'] # (boolean, represented as integer)

# List of all features; poi must be the first label - used later with targetFeatureSplit
features_list = poi + financial_features + email_features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# OPTIONAL: Transforming the dataset to pandas dataframe for easier handling
# data_df = pd.DataFrame.from_dict(data_dict, orient='index')
# data_df.replace(to_replace='NaN', value=np.nan, inplace=True)
    
## Print information about dataset
print "There are " + str(len(data_dict['SKILLING JEFFREY K'])) + " features in the dataset."
print "There are " + str(len(data_dict)) + " people in the dataset."

count = 0
for name in data_dict:
    if data_dict[name]['poi'] == True:
        count += 1
print "There are " + str(count) + " Person of Interest in the dataset."
    
    

###########################################
####### Step 2 : Remove Outliers ##########
###########################################

features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

# Plot bonus and salary to visualize any possible outliers
print "Plot before Outlier Removal:"
pyplot.xlabel("Bonus")
pyplot.ylabel("Salary")
for point in data:
    pyplot.scatter(point[0], point[1]) # (bonus, salary)
pyplot.show()


print "\n-> Outlier Value :" + str(data.max()) # The max value lies far away from other vales.


## Get outlier values
# Bonus Outlier
temp_max = 0
for name in data_dict:
    bonus = data_dict[name]['bonus'] 
    if bonus == 'NaN': 
        continue
    if temp_max < bonus: temp_max = bonus
# print temp


## 2 outliers are present in data
# 'TOTAL' is the accumulated value over all entries and hence should be removed
# 'THE TRAVEL AGENCY IN THE PARK' does not represent a person, hence should be removed
# 'LOCKHART EUGENE E' has all NaN values;does not contribute to the dataset. Hence removed.

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
for outlier in outliers:
    data_dict.pop(outlier, 0)

    
## Plot data after outlier removal
# A better visualisation of data can be observed afetr outlier removal
print " \nPlot after Outlier Removal:"
features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

pyplot.xlabel("Bonus")
pyplot.ylabel("Salary")
for point in data:
    pyplot.scatter(point[0], point[1]) # (bonus, salary)
pyplot.show()    



###########################################
####### Step 3 : Add new feature(s) #######
###########################################

## A important tool used for feature selection is feature_importances_
## -> use this to get an idea of how important a feature is
## get_features_ranking(data_dict, features_list)


## Compute importance of new features list for additional insights
## emails_to_poi_fraction ranks third. Interesting...
# get_features_ranking(data_dict, features_list)

for key in data_dict:
    data = data_dict[key]
    data["emails_from_poi_fraction"] = calculate_fraction(data['from_poi_to_this_person'], data['to_messages'])
    data["emails_to_poi_fraction"] = calculate_fraction(data['from_this_person_to_poi'], data['from_messages'])

    
# Remove extraneous features from original feature list
# Final Feature List
features_list = ['poi',
                 'exercised_stock_options',
                 'other',
                 'expenses',
                 'emails_to_poi_fraction',
                 'shared_receipt_with_poi',
                 'total_stock_value'
                ]

# Store to my_dataset for easy export below.
my_dataset = data_dict


## Go for selectKBest to pick high performing features
num_features = 4
select_k_best(data_dict, features_list, num_features)


# extract the features specified in features_list
data = featureFormat(my_dataset, features_list, sort_keys = True) # returns numpy array

# split into labels and features
# the first feature in the array is label thats why poi is put first
labels, features = targetFeatureSplit(data)

# Scale features using min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# Split data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)



#############################################################
######### Step 4 : Work with different classifiers ##########
#############################################################


clf_names = [
        "Naive Bayes",     
        "Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Decision Tree",
         "Random Forest", 
         "AdaBoost",          
         "Extra Trees"
        ]

classifiers = [
    GaussianNB(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=5, max_features=4, min_samples_split=2),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4, random_state=42),
    AdaBoostClassifier(algorithm='SAMME.R'),
    ExtraTreesClassifier(max_depth=5)
]

print "--------------------------------EVALUATING CLASSIFIERS---------------------------------"
 # iterate over classifiers
for name, clf in zip(clf_names, classifiers):
        clf.fit(features_train, labels_train)
        scores = clf.score(features_test,labels_test)
        print " "
        print "-> Classifier: " + name
        evaluate.evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print "\n-> Visualising Performance ----"
        plot_roc(clf, features_test, labels_test)
        print " "
        print "\nx------------------------------------------------------------------------------x"

print "** Using Precision,recall, accuracy and roc_curve; Decision Tree Classifier turns out \
        to be the best classifier.\n\n"



###########################################
###### Step 5 : Tune the Classifier #######
###########################################

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

dt_clf = DecisionTreeClassifier() 
parameters = {
    'max_depth': [1,2,3,4,5,6,8,9,10],
    'min_samples_split':[2,3,4,5],
    'min_samples_leaf':[1,2,3,4,5,6,7,8], 
    'criterion':('gini', 'entropy')
    }
grid_search = GridSearchCV(dt_clf, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
grid_search.fit(features, labels)
print 'Best Score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t %s: %r' % (param_name, best_parameters[param_name])
    
predictions = grid_search.predict(features_test)
print 'Accuracy:', accuracy_score(labels_test, predictions)
print 'Precision:', precision_score(labels_test, predictions)
print 'Recall:', recall_score(labels_test, predictions)



###########################################
###### Step 6 : Test the Classifier #######
###########################################

clf = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=4, min_samples_split=5)
clf.fit(features, labels)
plot_roc(clf, features_test, labels_test)


test_classifier(clf, my_dataset, features_list)

# Dump your classifier, dataset, and features_list so
# anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

