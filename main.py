from linkages import *
from clustering import AggloClustering
from utils import *

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

raw_data = read_csv("iris.csv")
X = raw_data.drop("class", axis=1).values
y = raw_data["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("single linkage...")
model = AggloClustering(single_linkage, 3)
model.fit(X_train)
y_pred = model.predict(X_test)
print("Our model \t: ", eval(y_test, y_pred))

legit_model = AgglomerativeClustering(n_clusters=3, linkage="single")
legit_model.fit(X_train)
centroids = find_cluster_centroids(X_train, legit_model.labels_)
y_pred = predict_on_centroids(X_test, centroids)
print("sklearn \t: ", eval(y_test, y_pred))

print("complete linkage...")
model = AggloClustering(complete_linkage, 3)
model.fit(X_train)
y_pred = model.predict(X_test)
print("Our model \t: ", eval(y_test, y_pred))

legit_model = AgglomerativeClustering(n_clusters=3, linkage="complete")
legit_model.fit(X_train)
centroids = find_cluster_centroids(X_train, legit_model.labels_)
y_pred = predict_on_centroids(X_test, centroids)
print("sklearn \t: ", eval(y_test, y_pred))

print("average linkage...")
model = AggloClustering(avg_linkage, 3)
model.fit(X_train)
y_pred = model.predict(X_test)
print("Our model \t: ", eval(y_test, y_pred))

legit_model = AgglomerativeClustering(n_clusters=3, linkage="average")
legit_model.fit(X_train)
centroids = find_cluster_centroids(X_train, legit_model.labels_)
y_pred = predict_on_centroids(X_test, centroids)
print("sklearn \t: ", eval(y_test, y_pred))

print("average group linkage...")
model = AggloClustering(avg_group_linkage, 3)
model.fit(X_train)
y_pred = model.predict(X_test)
print("Our model \t: ", eval(y_test, y_pred))