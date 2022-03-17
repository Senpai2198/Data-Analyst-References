###################################################################################################################################
############################################# Unsupervised Learning ###############################################################
###################################################################################################################################

#################################### K-Means Clustering #########################################################################

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
#plt.scatter(X[:, 0], X[:, 1], s=50);                             #!!Lines with single hashtag have to be input in the console!!

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

############################## Example

from sklearn.datasets import load_digits
digits = load_digits()
#digits.data.shape

#digits.data

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = clusters == i
    labels[mask] = mode(digits.target[mask])[0]
    
#np.zeros_like(clusters)

mask = clusters == 1
#print("which images are clustered into cluster 1?") 
#print(mask)
#print("")
#print("What is the length of mask? ")
#print(len(mask))
#print("")
#print("How many images are clustered into cluster 1?")
#print(mask.sum())

#print("Among the images that are clustered into cluster 1, what is the most possible digit?")
#print("")
#print(mode(digits.target[mask])[0])

from sklearn.metrics import accuracy_score
#accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for plot styling

mat = confusion_matrix(digits.target, labels)
#sns.heatmap(mat.T, square=True, annot=True, 
#            fmt='d', cbar=False,
#            xticklabels=digits.target_names,
#            yticklabels=digits.target_names)
#plt.xlabel('true label')
#plt.ylabel('predicted label');