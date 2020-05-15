import numpy as np
import pandas as pd
from mydict import dictionary
import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint


class Node:
    # Node class is meant to be used only by the
    # Tree class , so no more functions need to be
    # Added.
    def __init__(self, data=None):
        self.data = data
        self.right = None
        self.left = None

    def print_content(self):
        print("Data : " + str(self.data))
        print("left : " + str(self.left))
        print("right : " + str(self.right))


train_df = pd.read_csv("/sample_train.csv")
test_df = pd.read_csv("/sample_dev.csv")
test_df = test_df.drop("reviews.text", axis=1)
test_df = test_df.rename(columns={"rating": "label"})
train_df = train_df.drop("reviews.text", axis=1)
train_df = train_df.rename(columns={"rating": "label"})

data = train_df.values


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    data_equal = data[split_column_values == split_value]
    data_notequal = data[split_column_values != split_value]

    return data_equal, data_notequal


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values

    return potential_splits


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        # print(COLUMN_HEADERS[column_index], '-', len(np.unique(data[:, column_index])))
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value
# print(classify_data(train_df[train_df.contains_comfortable == 1].values))
data_eq, data_noteq = split_data(train_df.values, 0, 1)
print(get_potential_splits(test_df.values))

bestcolumn, bestvalue = determine_best_split(train_df.values,get_potential_splits(train_df.values))
print(bestvalue)


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

        # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return classification


    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        if len(potential_splits) == 0:
            classification = classify_data(data)
            return classification

        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        feature_name
        # question = "{} <= {}".format(feature_name, split_value)
        question = "{} = {}".format(feature_name, split_value)
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


# plt.show(sns.lmplot(x="contains_No", y="contains_Please",  data=test_df , hue="label",fit_reg= False,size = 20,aspect=1.5))