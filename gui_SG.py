import PySimpleGUI as sg
from IntegratedCode import calculate_accuracy,decision_tree_algorithm_with_nodes
from Tree_Node import Node, BinaryTree
import numpy as np
import pandas as pd
from mydict import dictionary


# Column layout
sg.theme('LightGreen3')
Review_text = ""
Train_path = ""
Test_path = ""
column1 = [[sg.Button('Write a review', font='Arial', size=(45, None), button_color=('white', '#3f3f44'), )],
           ]
column2 = [[sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Select train data', font='Arial', button_color=('white', '#3f3f44'), size=(16, None),
                          file_types=(("text files", ".csv"), ("all files", "*.*"),))],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Select test data', font='Arial', button_color=('white', '#3f3f44'), size=(16, None),
                          file_types=(("text files", ".csv"), ("all files", "*.*"),))],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.Button('Show accuracy', font='Arial', size=(16, None), button_color=('white', '#3f3f44'), )],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.Button('Classify', font='Arial', size=(16, None), button_color=('white', '#3f3f44'), )]]
column3 = [[sg.Multiline('This is the log', size=(45, 20), key='-OUTPUT-'+sg.WRITE_ONLY_KEY)],
           ]
column4 = [[sg.Button('Reset', font='Arial', size=(16, None), button_color=('black', '#fdcb9e'))],
           ]
column5 = [[sg.Button('Quit', font='Arial', size=(16, None), button_color=('black', '#fdcb9e'))],

           ]

layout = [[sg.Column(column1, justification='center')], [sg.Column(column2, justification='l'), sg.Column(column3)],
          [sg.Text(' ' * 10), sg.Column(column4, element_justification='left'), sg.Column(column5)],

          ]

# Display the window and get values

window = sg.Window('Decision Tree', layout, text_justification='left')

review_flag=0

while True:
    event, values = window.read()
    print(event, values)
    print(values)
    if event in (sg.WIN_CLOSED, 'Quit'):
        break
    elif event is 'Write a review':
        Review_text = sg.popup_get_text('Enter your review')
        rev=int(Review_text)

        arr=list(Review_text)
        print(type(arr[0]))
        print(arr)
        test2_df = pd.read_csv("input.csv")
        test2_df.iloc[0]=arr
        print(test2_df.iloc[0])
        print("done")
        review_flag=1

    elif event is 'Reset':
        print("Reset Triggered , do something !")

    elif event is 'Show accuracy':
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print(" ",text_color='black')
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print("Starting the execution",text_color='black')
        
        Train_path = list(values.values())[0]
        print("Train path : " + Train_path)
        Test_path = list(values.values())[1]
        print("Test_path :" + Test_path)
        
        test_df = pd.read_csv(Test_path)
        train_df = pd.read_csv(Train_path)
        train_df = train_df.drop("reviews.text", axis=1)
        train_df = train_df.rename(columns={"rating": "label"})
        test_df = test_df.drop("reviews.text", axis=1)
        test_df = test_df.rename(columns={"rating": "label"})
        
        TreeOfNodes =  BinaryTree()
        TreeOfNodes.root = decision_tree_algorithm_with_nodes(train_df, TreeOfNodes.root, max_depth=7)
        acc = calculate_accuracy(test_df,TreeOfNodes.root)
        print(acc)
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print("accuracy is:",text_color='black')
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print(acc,text_color='black')
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print("Execution Finished",text_color='black')
    
    elif event is 'Classify':
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print(" ",text_color='black')
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print("Starting the execution",text_color='black')

        if review_flag == 1: #if he entered a text
            print("here")
            Test_path="input.csv"
            Train_path="sample_train.csv"
            test_df = pd.read_csv(Test_path)
            train_df = pd.read_csv(Train_path)
            train_df = train_df.drop("reviews.text", axis=1)
            train_df = train_df.rename(columns={"rating": "label"})
            review_flag=0
        else:
            Train_path = list(values.values())[0]
            ("Train path : " + Train_path)
            Test_path = list(values.values())[1]
            print("Test_path :" + Test_path)

            test_df = pd.read_csv(Test_path)
            train_df = pd.read_csv(Train_path)
            train_df = train_df.drop("reviews.text", axis=1)
            train_df = train_df.rename(columns={"rating": "label"})
            test_df = test_df.drop("reviews.text", axis=1)
            test_df = test_df.rename(columns={"rating": "label"})
        
        TreeOfNodes =  BinaryTree()
        TreeOfNodes.root = decision_tree_algorithm_with_nodes(train_df, TreeOfNodes.root, max_depth=7)
        acc = calculate_accuracy(test_df,TreeOfNodes.root)
        print(acc)
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print("check classify.csv file",text_color='black')
        window['-OUTPUT-'+sg.WRITE_ONLY_KEY].print("Execution Finished",text_color='black')

    


window.close()
