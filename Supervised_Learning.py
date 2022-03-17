######################################### SUPERVISED LEARNING ###############################################################

import seaborn as sns
iris = sns.load_dataset('iris')
#import pandas as pd

X_iris = iris.drop('species', axis=1)  #Drop 'species' column
y_iris = iris['species']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)

################################### Gaussian Naive Bayes ####################################################################

from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data

from sklearn.metrics import accuracy_score
# accuracy_score(ytest, y_model)            #Enter this in console 

from sklearn.metrics import classification_report

#print(classification_report(ytest, y_model))

################################## k-NN Classifier ################################################################################

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(Xtrain, ytrain)

#knn.score(Xtest, ytest)

from sklearn.metrics import accuracy_score

y_model = knn.predict(Xtest) 
#accuracy_score(ytest, y_model)

#Pros

#One of the most attractive features of the K-nearest neighbor algorithm is that is simple to understand and easy to implement.

#Cons

#One of the obvious drawbacks of the KNN algorithm is the computationally expensive testing phase which is impractical in 
#industry settings. Furthermore, KNN can suffer from skewed class distributions. For example, if a certain class is very 
#frequent in the training set, it will tend to dominate the majority voting of the new example (large number = more common). 
#Finally, the accuracy of KNN can be severely degraded with high-dimension data because there is little difference between 
#the nearest and farthest neighbor.

##################################        SVM        ##############################################################################

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)


clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
#print('Breast cancer dataset')
#print('Accuracy of RBF SVC classifier on training set: {:.2f}'
#     .format(clf.score(X_train, y_train)))
#print('Accuracy of RBF SVC classifier on test set: {:.2f}'
#     .format(clf.score(X_test, y_test)))

(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

#X_cancer

#y_cancer

################## C-Parameter

clf_C = SVC(kernel='rbf', C=10000).fit(X_train, y_train)
#print('Breast cancer dataset')
#print('Accuracy of RBF SVC classifier on training set: {:.2f}'
#     .format(clf_C.score(X_train, y_train)))
#print('Accuracy of RBF SVC classifier on test set: {:.2f}'
#    .format(clf_C.score(X_test, y_test)))

#################MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_minmax = SVC(C=100).fit(X_train_scaled, y_train)
#print('Breast cancer dataset (normalized with MinMax scaling)')
#print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
#     .format(clf_minmax.score(X_train_scaled, y_train)))
#print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
#     .format(clf_minmax.score(X_test_scaled, y_test)))

##############################  Decision Tree  ##################################################################################

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)

clf.fit(Xtrain, ytrain)   

#tree.plot_tree(clf.fit(Xtrain, ytrain))

#clf.score(Xtest, ytest)

##################################################################################################################################





