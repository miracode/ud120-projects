#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(features_train, labels_train)
print "KNN Accuracy:", neigh.score(features_test, labels_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(features_train, labels_train)
print "Random Forest Accuracy:", rfc.score(features_test, labels_test)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc.fit(features_train, labels_train)
print "AdaBoost Accuracy:", abc.score(features_test, labels_test)


prettyPicture(neigh, features_test, labels_test, "neigh.png")
prettyPicture(rfc, features_test, labels_test, "rfc.png")
prettyPicture(abc, features_test, labels_test, "abc.png")

# for clf in [neigh, rfc, abc]:
#     try:
#         print "plotting"
#         prettyPicture(clf, features_test, labels_test)
#     except NameError:
#         print "passed"
#         pass
