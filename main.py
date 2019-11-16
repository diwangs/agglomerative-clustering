from linkages import avg_group_linkage
from clustering import AggloClustering

model = AggloClustering(avg_group_linkage, 2)
X = [[0.1, 0.1], [-0.1, -0.1], [0.0, 0.1], [2.1, 2.1], [1.9, 1.9]]
print(model.fit_predict(X))