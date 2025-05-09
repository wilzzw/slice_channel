import numpy as np

### Some array utilities ###

# Split input array a into equal shaped children arrays with overlapping array elements
def overlapping_split(a, num_element_childarray=2):
    # This will result in this number amount of child arrays:
    num_child_arrays = len(a) - num_element_childarray + 1
    output_child_set = []
    for i in range(num_child_arrays):
        if i+num_element_childarray >= len(a):
            output_child_set.append(a[i:])
        else:
            output_child_set.append(a[i:i+num_element_childarray])
    return output_child_set

def consecutive_split(a, diff=1):
    split_indices = np.argwhere(np.diff(a) > diff).flatten() + 1
    split_arrays = np.split(a, split_indices)
    return split_arrays

# Whether small_list is a sublist of big_list
# Easy with set operations
def is_sublist(small_list, big_list):
    for item in small_list:
        if item not in big_list:
            return False
    return True