from utils import *

class AggloClustering():
    def __init__(self, linkage_distance_function, n_clusters):
        self.linkage_distance_function = linkage_distance_function
        self.n_clusters = n_clusters
        self.subclusters = []

    def fit(self, X):
        # Turns X into list of tuples
        self.subclusters = list(map(tuple, X))
        while(len(self.subclusters) > self.n_clusters):
            # find pairs in subclusters with least distance
            nearest_pair_1, nearest_pair_2 = self.__find_nearest_pair()
            # Merge them as new subclusters
            self.subclusters.append([nearest_pair_1, nearest_pair_2])
            # Delete them from subclusters
            self.subclusters.remove(nearest_pair_1)
            self.subclusters.remove(nearest_pair_2)

    def fit_predict(self, X):
        self.fit(X)
        X = list(map(tuple, X))
        # Flatten cluster
        flattened_clusters = []
        for subcluster in self.subclusters:
            flattened_cluster = flatten(subcluster)
            flattened_clusters.append(flattened_cluster)
        # Make labels
        predictions = []
        for x in X:
            for cluster_idx in range(len(flattened_clusters)):
                if x in flattened_clusters[cluster_idx]:
                    predictions.append(cluster_idx)
                    break
        return predictions

    def predict(self, X):
        if not self.subclusters:
            raise ValueError("Train dulu...")
        # Calculate centroids
        centroids = []
        for subcluster in self.subclusters:
            flattened = flatten(subcluster)
            tupled = tuple(sum(elements)/len(flattened) for elements in zip(*flattened))
            centroids.append(tupled)
        # Predict clusters
        return predict_on_centroids(X, centroids)

    def __find_nearest_pair(self):
        if len(self.subclusters) < 2:
            raise ValueError("Yeet")
        else:
            best_min = float("inf")
            best_pair_1 = None
            best_pair_2 = None
            for i in range(len(self.subclusters) - 1):
                for j in range(i + 1, len(self.subclusters)):
                    curr_min = self.linkage_distance_function(self.subclusters[i], self.subclusters[j])
                    if best_min > curr_min:
                        best_min = curr_min
                        best_pair_1 = self.subclusters[i]
                        best_pair_2 = self.subclusters[j]
            return best_pair_1, best_pair_2