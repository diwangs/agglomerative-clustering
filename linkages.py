from utils import flatten, euclidean_distance

def avg_group_linkage(subcluster_1, subcluster_2):
    flattened_1 = flatten(subcluster_1)
    flattened_2 = flatten(subcluster_2)
    
    # find mean / centroid
    avg_1 = tuple(sum(elements)/len(flattened_1) for elements in zip(*flattened_1))
    avg_2 = tuple(sum(elements)/len(flattened_2) for elements in zip(*flattened_2))
    # calculate distance between centroid
    return euclidean_distance(avg_1, avg_2)

def complete_linkage(subcluster_1, subcluster_2):
    flattened_1 = flatten(subcluster_1)
    flattened_2 = flatten(subcluster_2)

    best_max = float("-inf")
    for x1 in flattened_1:
        for x2 in flattened_2:
            if best_max < euclidean_distance(x1, x2):
                best_max = euclidean_distance(x1, x2)
    return best_max

def single_linkage(subcluster_1, subcluster_2):
    flattened_1 = flatten(subcluster_1)
    flattened_2 = flatten(subcluster_2)

    best_min = float("inf")
    for x1 in flattened_1:
        for x2 in flattened_2:
            if best_min > euclidean_distance(x1, x2):
                best_min = euclidean_distance(x1, x2)
    return best_min

def avg_linkage(subcluster_1, subcluster_2):
    flattened_1 = flatten(subcluster_1)
    flattened_2 = flatten(subcluster_2)

    summed = sum([euclidean_distance(x1, x2) for x1 in flattened_1 for x2 in flattened_2])
    return summed / (len(flattened_1) * len(flattened_2)) # division by zero?
