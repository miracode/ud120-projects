#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# 1, 4
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print "Fit time: %s" %round(time() - t0, 3)

accuracy = clf.score(features_test, labels_test)
print "Accuracy: %s" % round(accuracy, 3)

# 2, 3
print "No. of features %s" % len(features_train[0])