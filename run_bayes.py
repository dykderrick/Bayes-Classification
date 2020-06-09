# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/9
# @Author  : Yingke Ding
# @File    : run_bayes.py
# @Software: PyCharm
from bayes_classifier import NaiveBayesClassifier
from train_test_division import get_train_test_dataset


def _evaluation(confusion_matrix):
    """
    Using "Accuracy", "Error rate", "Precision" and "Recall" metrics to evaluate the result.
    :param confusion_matrix: Will be a 3 by 3 list of list specifically for Iris dataset.
    :return: None
    """

    # Compute 3x3 confusion matrix to get accuracy and error rate.
    #
    # Perfect trick of using zip to deal with a list of list.
    # Get idea from https://stackoverflow.com/questions/13783315/sum-of-list-of-lists-returns-sum-list
    hit_count = sum(matrix_row[index] for index, matrix_row in enumerate(confusion_matrix))
    all_count = sum([sum(i) for i in zip(*confusion_matrix)])
    accuracy = hit_count / all_count
    print("Accuracy:   " + "%.2f" % (100 * accuracy) + "%")
    print("Error rate: " + "%.2f" % (100 * (1 - accuracy)) + "%")

    # Compute 3x3 confusion matrix to get precision and recall rate for each kind of class.
    for i in range(len(confusion_matrix)):
        hit_count = confusion_matrix[i][i]
        precision_count = sum(counts[i] for counts in confusion_matrix)
        recall_count = sum(confusion_matrix[i])
        precision_rate = hit_count / precision_count
        recall_rate = hit_count / recall_count
        print("Class " + str(i) + " precision rate: " + "%.2f" % (100 * precision_rate) + "%. " +
              "Recall rate: " + "%.2f" % (100 * recall_rate) + "%.")
    print("\n")


def _get_a_test_tuple_result(index, test_tuple, all_train_set, index2name, confusion_matrix):
    """
    Invoke :func:`bayes_classifier.NaiveBayesClassifier`
    to compute bayes classification result for a specific test tuple.
    Also print result information.
    :param index: test number
    :param test_tuple: attributes and actual class info
    :param all_train_set: train set
    :param index2name: class index to name
    :param confusion_matrix: to keep record for evaluation
    :return: None
    """

    print("# " + str(index))
    predict_result = NaiveBayesClassifier(test_tuple, all_train_set, index2name).get_predict_result()
    print("[TEST      TUPLE]        " + str(test_tuple))
    print("[ACTUAL    CLASS]        " + "class " + str(test_tuple["class"]))
    print("[EACH CLASS PROB]        " + str(predict_result))

    max_prob = max([prob["prob"] for prob in predict_result])
    print("[MAX        PROB]        " + str(max_prob))

    predict_class = [prob["predict_class"] for prob in predict_result if prob["prob"] == max_prob][0]
    print("[PREDICT   CLASS]        " + "class " + str(predict_class))

    confusion_matrix[predict_class][test_tuple["class"]] += 1
    if predict_class == test_tuple["class"]:
        is_hit = True
    else:
        is_hit = False

    print("[IS      MATCHES]        " + str(is_hit))
    print("\n")


def main():
    """
    Main program to see algorithm result by default.
    :return: None
    """

    # Get data
    dataset_path = "./dataset/Iris.csv"
    name2index, index2name, all_train_set, all_test_set = get_train_test_dataset(dataset_path)

    """
              | Actual 0  |  1  |  2  |
    -----------------------------------
    predict 0 |    hit    |     |     |
    -----------------------------------
    predict 1 |           | hit |     |
    -----------------------------------
    predict 2 |           |     | hit |
    """
    confusion_matrix = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]

    # Start test for each test tuple
    for index, test_tuple in enumerate(all_test_set):
        _get_a_test_tuple_result(index, test_tuple, all_train_set, index2name, confusion_matrix)

    # Evaluate
    _evaluation(confusion_matrix)


if __name__ == '__main__':
    main()
