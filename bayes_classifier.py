# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 3:09 下午
# @Author  : Yingke Ding
# @FileName: bayes_classifier.py
# @Software: PyCharm


def _compute_conditional_prob(class_data, attribute_value, attribute_index):
    """
    Compute a specific attribute's conditional probability of a given kind of class.
    :param class_data: all train set data for a specific kind of class.
    :param attribute_value: just value for a feature of iris.
    :param attribute_index: index.
    :return: conditional probability stored as a list. e.g. 4/45 will be returned as [4, 45]
    """
    attribute_occurrences = 0
    all_count = len(class_data)

    for data_tuple in class_data:
        if float(attribute_value) == float(data_tuple["attributes"][attribute_index]):
            attribute_occurrences += 1

    return [attribute_occurrences, all_count]


def _laplacian_correction(probs):
    """
    Avoid zero-prob problem.
    Adding 1 to each case if any one of conditional probabilities is zero.
    :param probs: conditional probs list.
    :return: Laplacian-corrected conditional probs.
    """
    return [[prob[0] + 1, prob[1] + len(probs)] for prob in probs]


def _calculate_prob(probs):
    """
    Computes conditional probs multiplication based on a suppose that each condition is non-independent.
    :param probs: conditional-probs list.
    :return: multiplication of each prob.
    """
    prob_multiplication = 1.
    for prob in probs:
        prob_multiplication *= prob[0]
        prob_multiplication /= prob[1]

    return prob_multiplication


class NaiveBayesClassifier:
    """
    Algorithm class.
    Each time a tuple in test set together with the train set will be parsed in to perform optimizing.
    """

    def __init__(self, test_tuple, train_data, index2name):
        """
        Defines class variables.
        Invoke two class methods to perform the algorithm.
        :param test_tuple: A specific tuple in test set. Stored in a dictionary data structure.
        :param train_data: All train set tuples.
        :param index2name: class index to name dictionary.
        """
        self.test_tuple = test_tuple
        self.train_data = train_data
        self.index2name = index2name
        self.predict_result = []  # to be returned

        self._set_class_probabilities()
        self._algorithm()  # Invoke algorithm

    def _set_class_probabilities(self):
        """
        Computes probabilities for each kind of classes.
        But actually this doesn't make any influences on the final result,
        Every kind of Iris actually gets 1/3 class_prob.
        :return: probs list.
        """
        self._class_probabilities = {key: 0 for key in self.index2name}

        for data_dict in self.train_data:
            self._class_probabilities[data_dict["class"]] += 1

        self._class_probabilities = {key: (self._class_probabilities[key] / len(self.train_data)) for key in
                                     self._class_probabilities.keys()}

    def _compute_test_conditional_probs(self, class_data):
        """
        Computes each attribute's conditional prob for a specific test tuple.
        :param class_data: is all "conditional-class" train set tuples.
        :return: a list of list showing each attribute's probs. Each probability is stored as a list.
        For example: 3/45 is stored as [3, 45].
        """
        probs = []

        for attribute_index, attribute_value in enumerate(self.test_tuple["attributes"]):
            probs.append(_compute_conditional_prob(class_data, attribute_value, attribute_index))

        for prob in probs:
            if prob[0] == 0:
                probs = _laplacian_correction(probs)
                break

        return probs

    def _algorithm(self):
        """
        Core part for Naive Bayes Algorithm.
        Keep record of probabilities into self.predict_result for each possible class of a specific test tuple.
        """
        for key in self.index2name.keys():
            conditional_probs = self._compute_test_conditional_probs([class_data for class_data in self.train_data
                                                                      if class_data["class"] == key])
            final_prob = _calculate_prob(conditional_probs) * self._class_probabilities[key]

            this_dict = dict()
            this_dict["prob"] = final_prob
            this_dict["predict_class"] = key
            self.predict_result.append(this_dict)

    def get_predict_result(self):
        """
        Result getter.
        :return: result.
        """
        return self.predict_result
