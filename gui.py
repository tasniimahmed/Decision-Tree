from tkinter import *
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from IntegratedCode import calculate_accuracy,decision_tree_algorithm_with_nodes,train_df
from Tree_Node import Node, BinaryTree
import numpy as np
import pandas as pd
from mydict import dictionary


def sel_file():

    file = filedialog.askopenfile(initialdir="/", title="select file",
                                    filetypes=(("text files", ".csv"), ("all files", "*.*")))
    print(file)
    test_df = pd.read_csv(file)
    test_df = test_df.drop("reviews.text", axis=1)
    test_df = test_df.rename(columns={"rating": "label"})
    TreeOfNodes =  BinaryTree()
    #tree = decision_tree_algorithm_without_nodes(train_df,  max_depth=5)
    TreeOfNodes.root = decision_tree_algorithm_with_nodes(train_df, TreeOfNodes.root, max_depth=7)
    ##print(TreeOfNodes.root)
    acc = calculate_accuracy(test_df,TreeOfNodes.root)
    acclabel= Label(root,text=acc)
    acclabel.place(relx = 0.5,  
                   rely = 0.5, 
                   anchor = 'center') 




root = Tk()
selbutton =Button(root, text="Select Sample",command=sel_file)



#acclabel.pack()
selbutton.pack()
root.geometry("1000x500")
root.mainloop()

