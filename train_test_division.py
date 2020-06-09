# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 11:15 上午
# @Author  : Yingke Ding
# @FileName: train_test_division.py
# @Software: PyCharm
import csv


def get_data(csv_file_path):
    """
    Basic data reading method.
    Will load Iris data into "data",
    and also returns a "class name - integer" mapping to replace Iris-setosa, Iris-versicolor and Iris-virginica.
    :param csv_file_path: particularly it will be a relative path "./dataset/Iris.csv".
    :return: name2index, index2name, data
    """
    name2index = dict()
    data = []

    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)

        names = []
        for row in reader:
            if row[-1] not in names:
                names.append(row[-1])

            this_dict = {"attributes": row[:-1], "class": len(names) - 1}  # Use a dictionary to record raw data classes
            data.append(this_dict)

        for index, value in enumerate(names):
            name2index[value] = index

    index2name = {value: key for key, value in name2index.items()}  # reverse

    return name2index, index2name, data


def split_dataset(class_data, percentage):
    """
    Split a particular class of data (e.g. all Iris-setosa tuples)
    into train set (set for Bayes algorithm to calculate probabilities)
    and test set (set for running Bayes algorithm to make a comparision between given class and Bayes optimized class).
    :param class_data: a list of a specific kind of class of data
    :param percentage: the length ratio of train set and test set.
    Default value will be 0.9 (i.e. 45 train tuples and 5 test tuples for each class of data).
    :return: train_set, test_set
    """
    train_length = int(len(class_data) * percentage)

    train_set = class_data[:train_length]
    test_set = class_data[train_length:]  # actually not the complete precise ratio but we have to keep all the data

    return train_set, test_set


def get_train_test_dataset(csv_file_path, percentage=0.9):
    """
    General method for dataset reading procedure.
    Will call :func:`train_test_division.get_data` and
    :func:`train_test_division.split_dataset()` to perform general operations on dataset.
    :param csv_file_path: Will be a String of "./dataset/Iris.csv"
    :param percentage: the length ratio of train set and test set.
    :return: name2index, index2name, all_train_set, all_test_set
    """
    name2index, index2name, data = get_data(csv_file_path)

    all_train_set = []
    all_test_set = []

    for key in index2name.keys():
        class_train_set, class_test_set = split_dataset([row for row in data if row["class"] == key], percentage)
        all_train_set += class_train_set
        all_test_set += class_test_set

    return name2index, index2name, all_train_set, all_test_set
