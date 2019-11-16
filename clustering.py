from utils import flatten, euclidean_distance

class AggloClustering():
    def __init__(self, distance_function, n_clusters):
        self.distance_function = distance_function
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
        predictions = []
        for x in X:
            best_distance = float("inf") 
            best_cluster = None
            for cluster_idx in range(len(centroids)):
                distance = euclidean_distance(x, centroids[cluster_idx])
                if best_distance > distance:
                    best_cluster = cluster_idx
                    best_distance = distance
            predictions.append(best_cluster)
        return predictions

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def __find_nearest_pair(self):
        if len(self.subclusters) < 2:
            raise ValueError("Yeet")
        else:
            best_min = float("inf")
            best_pair_1 = None
            best_pair_2 = None
            for i in range(len(self.subclusters) - 1):
                for j in range(i + 1, len(self.subclusters)):
                    curr_min = self.distance_function(self.subclusters[i], self.subclusters[j])
                    if best_min > curr_min:
                        best_min = curr_min
                        best_pair_1 = self.subclusters[i]
                        best_pair_2 = self.subclusters[j]
            return best_pair_1, best_pair_2