"""
In this File, we shall be implementing the K-Nearest Neighbour
Algorithm from Scratch.

Algorithm Basics

We collect the data in an N Dimensional Format

Then we also get the test data, for which we need to find out
what classes do these tests points belong to.

What we can do is that for each point in the test data,get the distance
of this test point from every point in the training data.

Then based on the K value we have, we find the top K simillar
points to this test point and assign the class based on the majority of 
the classes
"""

from numpy import *
import operator
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1], [5, 4], [6, 10]])
    labels = ["A", "A", "B", "B", "A", "B"]

    return group, labels


def classify(point, data, labels, k):
    # Take Distance Between Point and All data points
    distance_point = [
        (label, np.linalg.norm(item - point, ord=2))
        for item, label in zip(data, labels)
    ]

    # Take Top K Distances
    distance_point.sort(key=lambda x: x[1])
    print(distance_point[:k])
    # Get Lables for Corresponding Points
    top_labels = [label[0] for label in distance_point[:k]]
    print(top_labels)
    new_dict = Counter(top_labels)
    print(new_dict.items())
    # Get Sorted Labels based on Distance
    new_dict_sorted = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Return the Majority Class Label
    return new_dict_sorted[0][0]


group, labels = createDataSet()
print(group)
# print(list_x, list_y)
# Simple Plot Of the Points that we  have
# print([x for x in labels], [(x, y) for x, y in zip(list_x, list_y)])
# Multiple Annotation is some trickery in Matplotlib
# plt.scatter(x=list_x, y=list_y)
# plt.annotate("A", (1, 1.1))
# plt.annotate("A", (1, 1.0))
# plt.annotate("B", (0, 0))
# plt.annotate("B", (0, 0.1))
# plt.show()
point = array([[2, 0.3]])
new_label = classify(point, group, labels, k=6)
print(new_label)
