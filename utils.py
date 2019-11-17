def euclidean_distance(tuple_1, tuple_2):
    tuple_dist = tuple(map(lambda x, y: x - y, tuple_1, tuple_2))
    return sum([x**2 for x in tuple_dist])**0.5

def flatten(input_array):
    result_array = []

    if isinstance(input_array, tuple):
        result_array.append(input_array)
        return result_array

    for element in input_array:
        if isinstance(element, tuple):
            result_array.append(element)
        elif isinstance(element, list):
            result_array += flatten(element)
    return result_array

def find_cluster_centroids(X, y):
    assert len(X) == len(y)
    # Get array of elements for each cluster
    clusters = set(y)
    clustered_data = []
    for cluster in clusters:
        cluster_data = []
        for i in range(len(y)):
            if y[i] == cluster:
                cluster_data.append(X[i])
        clustered_data.append(cluster_data)
    
    # Compute centroid of each clusters
    centroids = []
    for cluster_data in clustered_data:
        tupled = tuple(sum(elements)/len(cluster_data) for elements in zip(*cluster_data))
        centroids.append(tupled)
    return centroids

def predict_on_centroids(X, centroids):
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

def eval(y_gold, y_pred):
        assert len(y_gold) == len(y_pred)
        classes = set(y_gold)
        clusters = set(y_pred)
        
        # compute confusion matrix
        confusion_mtx = {}
        for klass in classes:
            cluster_membership = {}
            for cluster in clusters:
                cluster_membership[cluster] = sum([y1 == klass and y2 == cluster for y1, y2 in zip(y_gold, y_pred)])
            confusion_mtx[klass] = cluster_membership
        
        # compute accuracy, based on dominant cluster in each class
        dominant_cluster = {}
        for klass in classes:
            dominant_cluster[klass] = max(confusion_mtx[klass], key=confusion_mtx[klass].get)
        true = sum([confusion_mtx[klass][dominant_cluster[klass]] for klass in classes])
        total = sum([sum(confusion_mtx[klass].values()) for klass in classes])
        return true/total
