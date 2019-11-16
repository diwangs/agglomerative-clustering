from utils import flatten, euclidean_distance

def avg_group_linkage(subcluster_1, subcluster_2):
    flattened_1 = flatten(subcluster_1)
    flattened_2 = flatten(subcluster_2)
    # find mean / centroid
    tuple_1 = tuple(sum(elements)/len(flattened_1) for elements in zip(*flattened_1))
    tuple_2 = tuple(sum(elements)/len(flattened_2) for elements in zip(*flattened_2))
    # calculate distance between centroid
    return euclidean_distance(tuple_1, tuple_2)