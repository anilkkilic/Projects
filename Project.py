import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.datasets import load_wine
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# print(dir(datasets))
wine = load_wine()
# print(wine.DESCR)

print(wine.feature_names)
# print(wine.data)
print(wine.data.shape)
print(wine.target_names)
df = pd.DataFrame(wine.data)
df.columns = wine['feature_names']
print(df)

X = wine['data']
y = wine['target']

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# --------------------------K-Means--------------------------------
kmeans = KMeans(n_clusters=3)
K_model = kmeans.fit(wine.data)
# print(K_model)

labels_knn = K_model.labels_
predict_1 = K_model.fit_predict(wine.data)
# print(labels_knn)
# print(predict_1)

centroids = K_model.cluster_centers_
# print(centroids)
# 1st, 2nd, 3rd cluster's centroids.

df['target'] = wine['target']
df['K-Means predicted'] = predict_1
df = df.loc[:, ['target', 'K-Means predicted']]
print(df.head(20))

# Adjusted Rand Score
print("Adjusted Rand Score of K-Means", adjusted_rand_score(df['target'], predict_1))

# Homogeneity Score
print("Homogeneity Score of K-Means", homogeneity_score(df['target'], predict_1))

# Completeness Score
print("Completeness Score of K-Means", completeness_score(df['target'], predict_1))

# V-Measure Score
print("V-Measure Score of K-Means", v_measure_score(df['target'], df['K-Means predicted']))

# -------------------Mean Shift---------------------

ms = MeanShift()
ms.fit(wine.data)
# print(ms.fit(wine.data))
labels_mean = ms.labels_
# print(labels_mean)
predict_2 = ms.fit_predict(wine.data)
cluster_centers = ms.cluster_centers_
# print(cluster_centers)

# Adjusted Rand Score
print("Adjusted Rand Score of Mean Shift:", adjusted_rand_score(df['target'], predict_2))

# Homogeneity Score
print("Homogeneity Score of Mean Shift:", homogeneity_score(df['target'], predict_2))

# Completeness Score
print("Completeness Score of Mean Shift:", completeness_score(df['target'], predict_2))

# V-Measure Score
print("V-Measure Score of Mean Shift", v_measure_score(df['target'], predict_2))

n_clusters_ = len(np.unique(df['target']))
colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
for i in range(len(wine.data)):
    plt.plot(X[i][0], X[i][1], colors[labels_mean[i]], markersize=3)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker=".", color='k', s=20, linewidths=5, zorder=10)
plt.show()


# -------------------------K-Nearest Neighbors (KNN)-------------------------

knn = KNeighborsClassifier(n_neighbors=5)
# print(knn)
knn.fit(X_train, y_train)
# print(knn.fit(X_train, y_train))
y_pred_1 = knn.predict(X_test)
print("Accuracy of the KNN is : ", metrics.accuracy_score(y_test, y_pred_1))

# K-Fold Cross Validation (K=10)
scores_knn = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print("K-Fold Cross Validation Score of KNN:", scores_knn)

# balanced_accuracy_score
print("Balanced Accuracy Score of KNN:", balanced_accuracy_score(y_test, y_pred_1))

# f1-Score
print("F-1 Score of KNN:", f1_score(y_test, y_pred_1, average='weighted'))

# ROC AUC Score
y_pred_proba_1 = knn.predict_proba(X_test)
print("ROC AUC Score of KNN:", roc_auc_score(y_test, y_pred_proba_1, multi_class='ovr'))

# ROC Curve
knn_fpr, knn_tpr, _ = metrics.roc_curve(y_test, y_pred_1, pos_label=2)

# -------------------------SVM (Support Vector Machine)-----------------

clf = svm.SVC(kernel='linear', probability=True)
# print(clf)

clf.fit(X_train, y_train)
y_pred_2 = clf.predict(X_test)
print("Accuracy of the SVM is : ", metrics.accuracy_score(y_test, y_pred_2))

print(classification_report(y_test, y_pred_2))

# K-Fold Cross Validation (K=10)
scores_svm = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print("K-Fold Cross Validation Scores of SVM:", scores_svm)

# balanced_accuracy_score
print("Balanced Accuracy Score of SVM", balanced_accuracy_score(y_test, y_pred_2))

# f1-Score
print("F-1 Score of SVM", f1_score(y_test, y_pred_2, average='weighted'))

# ROC AUC Score
y_pred_proba_2 = clf.predict_proba(X_test)
print("ROC AUC Score of SVM:", roc_auc_score(y_test, y_pred_proba_2, multi_class='ovr'))

# ROC Curve
svm_fpr, svm_tpr, _ = metrics.roc_curve(y_test, y_pred_2, pos_label=2)

plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN')
plt.plot(svm_fpr, svm_tpr, marker='*', label='SVM')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
