# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10
# @Author  : Yingke Ding
# @File    : run_different_division.py
# @Software: PyCharm
from bayes_classifier import NaiveBayesClassifier
from train_test_division import get_train_test_dataset


def main():
    """
    Main program to test accuracy under different train-test set length ratio.
    :return: None
    """
    # Get data
    dataset_path = "./dataset/Iris.csv"
    percentages = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
    for percentage in percentages:
        name2index, index2name, all_train_set, all_test_set = get_train_test_dataset(dataset_path, percentage)
        confusion_matrix = [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]

        # Start test for each test tuple
        for test_tuple in all_test_set:
            predict_result = NaiveBayesClassifier(test_tuple, all_train_set, index2name).get_predict_result()
            max_prob = max([prob["prob"] for prob in predict_result])
            predict_class = [prob["predict_class"] for prob in predict_result if prob["prob"] == max_prob][0]
            confusion_matrix[predict_class][test_tuple["class"]] += 1

        hit_count = sum(matrix_row[index] for index, matrix_row in enumerate(confusion_matrix))
        all_count = sum([sum(i) for i in zip(*confusion_matrix)])
        accuracy = hit_count / all_count
        print("Percentage:" + str(percentage) + ". Accuracy:   " + "%.2f" % (100 * accuracy) + "%")


if __name__ == '__main__':
    main()
