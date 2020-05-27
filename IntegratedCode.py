import numpy as np
import pandas as pd
import io
from mydict import dictionary
# import matplotlib.pyplot as plt
# import seaborn as sns
from Tree_Node import Node, BinaryTree
import random
from pprint import pprint
import pydot
import os
# from PIL import Image
import graphviz
from graphviz import Source
Tree_plot = graphviz.Digraph('Tree',format='png')

G = pydot.Dot(graph_type="digraph")

# train_df = pd.read_csv("sample_train.csv")
# test_df = pd.read_csv("sample_dev.csv")
# test_df = test_df.drop("reviews.text", axis=1)
# test_df = test_df.rename(columns={"rating": "label"})
# train_df = train_df.drop("reviews.text", axis=1)
# train_df = train_df.rename(columns={"rating": "label"})

# data = train_df.values


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
    # return positive or negative based on majority
    classification = unique_classes[index]

    return classification


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    data_equal = data[split_column_values == split_value]
    data_notequal = data[split_column_values != split_value]

    return data_equal, data_notequal


def get_potential_splits(data):
    potential_splits = dictionary()
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        potential_splits.add(column_index, unique_values)

    return potential_splits


def get_entropy(data):
    # sum of prob of all classes * -log(prob)
    rate_col = data[:, -1]
    # calculate the number of each class in the given data after that we have to calc the probab by dividing by the sum of two classes
    _, num = np.unique(rate_col, return_counts=True)
    prob = num / num.sum()  # we can add counts[0] + count[1] but this may give error if a class isn't found at all
    entropy = sum(prob * -np.log2(prob))
    return entropy


def get_overall_entropy(data_equal, data_not_equal):
    data_point = len(data_equal) + len(data_not_equal)
    p_data_equal = len(data_equal) / data_point
    p_data_not_equal = len(data_not_equal) / data_point
    overall_entropy = (p_data_equal * get_entropy(data_equal) + p_data_not_equal * get_entropy(data_not_equal))

    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        # print(COLUMN_HEADERS[column_index], '-', len(np.unique(data[:, column_index])))
        for value in potential_splits.get(column_index):
            data_equal, data_not_equal = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = get_overall_entropy(data_equal, data_not_equal)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


# print(classify_data(train_df[train_df.contains_comfortable == 1].values))
###data_eq, data_not_eq = split_data(train_df.values, 0, 1)
# print(get_potential_splits(test_df.values))

###bestcolumn, bestvalue = determine_best_split(train_df.values, get_potential_splits(train_df.values))

# print(bestcolumn)

###TreeOfNodes = BinaryTree()


def decision_tree_algorithm_with_nodes(df, current_node, counter=0, min_samples=2, max_depth=5):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS

        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

        # base cases
        # len of data returns the total number of features
        # upon reaching 1 feature i will break of the loop
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        current_node = Node(classification)
        return current_node


    # recursive part
    else:
        counter = counter + 1

        # helper functions
        potential_splits = get_potential_splits(data)
        # get potential splits return a dicitonar of columns with unique values in each column
        if potential_splits.counter == 0:
            classification = classify_data(data)
            current_node = Node(classification)
            print(current_node.value)
            return current_node

        split_column, split_value = determine_best_split(data, potential_splits)
        data_equal, data_not_equal = split_data(data, split_column, split_value)

        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        if counter == 1:
            current_node = Node(feature_name)

        # find answers (recursion)

        if current_node is None:
            current_node = Node(feature_name)
        current_node.right = decision_tree_algorithm_with_nodes(data_equal, current_node.right, counter, min_samples,
                                                                max_depth)
        current_node.left = decision_tree_algorithm_with_nodes(data_not_equal, current_node.left, counter, min_samples,
                                                               max_depth)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if current_node.right == current_node.left:
            current_node.value = current_node.right.value

        # else:
        #     TreeOfNodes.insert(yes_node,'yes')
        #     TreeOfNodes.insert(no_node, 'no')

        return current_node


###TreeOfNodes.root
###TreeOfNodes.root = decision_tree_algorithm_with_nodes(train_df, TreeOfNodes.root, max_depth=5)
# print("Column headers" + str(len(COLUMN_HEADERS)))
# print(COLUMN_HEADERS.values)



# pprint(tree)
# print("                         ********** TREE *************")
# print(TreeOfNodes.root)

# decision_tree_algorithm_with_nodes(train_df, TreeOfNodes, max_depth=5)
# print("The value of root : ")
# print(TreeOfNodes.root)
#
# print('finished training')


first_time = 1
my_path= []
#classify without drawing
def classify_example_with_Nodes(example,tree):

    if tree.left == None and tree.right == None:
        #base case of classification +ve or -Ve
        print(tree.value)
        my_path.append(tree.value)
        return tree.value
        
    else:
        #it's a question 
        if example[tree.value] == 1: #a no answer
            print(tree.value)
            my_path.append(tree.value)
            return classify_example_with_Nodes(example,tree.right)
        
        elif example[tree.value]==0: #a yes answer
            print(tree.value)
            my_path.append(tree.value)
            return classify_example_with_Nodes(example,tree.left)

#Tasnim commented this
# def classify_example_with_Nodes(example, tree, first_time):
#     # first_time_local = first_time
#     global my_node

#     if tree.left == None and tree.right == None:
#         # base case of classification +ve or -Ve
#         # print("here")
#         new_node = pydot.Node(tree.value, style="filled", fillcolor="blue")
#         G.add_node(new_node)
#         edge = pydot.Edge(my_node, new_node)
#         my_node = new_node
#         G.add_edge(edge)
#         return tree.value

#     else:
#         # it's a question
#         if example[tree.value] == 1:  # a yes answer

#             if first_time == 1:
#                 # print("hello1")
#                 my_node = pydot.Node(tree.value, style="filled", fillcolor="blue")
#                 G.add_node(my_node)
#                 # print("************yes*************")
#                 first_time = 0
#             else:
#                 # print("hello2")
#                 new_node = pydot.Node(tree.value, style="filled", fillcolor="blue")
#                 G.add_node(new_node)
#                 edge = pydot.Edge(my_node, new_node, label="yes")
#                 my_node = new_node
#                 G.add_edge(edge)
#             # return tree.left.value
#             return classify_example_with_Nodes(example, tree.right, first_time)

#         elif example[tree.value] == 0:  # a no answer
#             if first_time == 1:
#                 # print("hello1 no")
#                 my_node = pydot.Node(tree.value, style="filled", fillcolor="blue")
#                 G.add_node(my_node)
#                 # print("************no*************")
#                 first_time = 0
#             else:
#                 # print("hello2 no")
#                 new_node = pydot.Node(tree.value, style="filled", fillcolor="blue")
#                 G.add_node(new_node)
#                 edge = pydot.Edge(my_node, new_node, label="no")
#                 my_node = new_node
#                 G.add_edge(edge)
#             # return tree.right.value
#             # print(tree.right.value)
#             return classify_example_with_Nodes(example, tree.left, first_time)


# example=test_df.iloc[0] #first row of test_df
# x=classify_example_with_Nodes(example,TreeOfNodes.root,first_time)

# print(x)


def calculate_accuracy(df, tree):
    #Tasnim commented this
    #df["classification"] = df.apply(classify_example_with_Nodes, axis=1, args=(tree, 1,))
    #applying the classification function on the given df
    df["classification"] = df.apply(classify_example_with_Nodes, axis=1, args=(tree,))
    #writing the result of classification in a file
    df["classification"].to_csv('classify.csv', encoding='utf-8')
    #check if we want to calculate the accuracy or not (just printing in a file) by checking if the label column exists or not
    if 'label' in df:
        df["classification_correct"] = df["classification"] == df["label"]
        accuracy = df["classification_correct"].mean()
        return accuracy


###accuracy = calculate_accuracy(test_df, TreeOfNodes.root)
###print(accuracy)

# def draw_tree(count = 0,node = TreeOfNodes.root):
#     count = count +1
#     if node.right is not None or node.left is not None:
#         if node.right.value == node.left.value:
#             Tree_plot.node(node.right.value, node.right.value)
#             Tree_plot.edge(node.value, node.right.value)
#         elif node.right.value =='Positive' or node.left.value == 'Positive' or node.right.value =='Negative' or node.left.value == 'Negative':
#             Tree_plot.node(node.value,node.value)
#             Tree_plot.node(node.right.value+'right',node.right.value)
#             Tree_plot.node(node.left.value+'left', node.left.value)
#             Tree_plot.edge(node.value,node.right.value+'right')
#             Tree_plot.edge(node.value,node.left.value+'left')
#         else:
#             Tree_plot.node(node.value,node.value)
#             Tree_plot.node(node.right.value,node.right.value)
#             Tree_plot.node(node.left.value, node.left.value)
#             Tree_plot.edge(node.value,node.right.value)
#             Tree_plot.edge(node.value,node.left.value)
#             draw_tree(count,node = node.left)
#             draw_tree(count,node = node.right)



#Tasnim Commented this
# def draw_tree(leftnode = TreeOfNodes.root.left, node = TreeOfNodes.root, rightnode = TreeOfNodes.root.right):


#     if node.right is not None or node.left is not None:
#         if node.right.value == node.left.value:
#             # print(node.right.value)
#             Tree_plot.node(node.right.value+node.value, node.right.value)
#             Tree_plot.edge(node.value, node.right.value+node.value, label="Yes or No")
#         else:
#             # print('contains_'+node.value.split('contains_')[1])
#             Tree_plot.node(node.value,'contains_'+node.value.split('contains_')[1])
#             Tree_plot.node(node.left.value + node.value, node.left.value)
#             Tree_plot.node(node.right.value + node.value ,node.right.value)

#             Tree_plot.edge(node.value,node.right.value + node.value,label="Yes")
#             Tree_plot.edge(node.value,node.left.value + node.value,label= "No")
#             node.left.value = node.left.value + node.value
#             node.right.value = node.right.value + node.value
#             draw_tree(leftnode = leftnode.left, node= leftnode,rightnode = leftnode.right)
#             draw_tree(leftnode = rightnode.left, node= rightnode,rightnode = rightnode.right)
#     # elif node.right is not None:
#     #     # continue tracing this node
#     # elif node.left is not None:
#     #     #continue tracing
# draw_tree()
# G.write_png('example1_graph.png')

# Tree_plot.render()


# print(calculate_accuracy(test_df, tree))
# plt.show(sns.lmplot(x="contains_No", y="contains_Please",  data=test_df , hue="label",fit_reg= False,size = 20,aspect=1.5))
