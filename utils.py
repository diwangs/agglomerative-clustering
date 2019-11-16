def euclidean_distance(tuple_1, tuple_2):
    tuple_dist = tuple(map(lambda x, y: x - y, tuple_1, tuple_2))
    return sum([x**2 for x in tuple_dist])**0.5

def flatten(input_array):
    result_array = []
    for element in input_array:
        if isinstance(element, tuple):
            result_array.append(element)
        elif isinstance(element, list):
            result_array += flatten(element)
    return result_array