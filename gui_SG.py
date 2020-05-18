import PySimpleGUI as sg

# Column layout
sg.theme('Black')
Review_text = ""
Train_path = ""
Test_path = ""
column1 = [[sg.Button('Write a review', font='Arial', size=(45, None), button_color=('white', '#005800'), )],
           ]
column2 = [[sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Select train data', font='Arial', button_color=('white', '#005800'), size=(16, None),
                          file_types=(("text files", ".csv"), ("all files", "*.*"),))],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Select test data', font='Arial', button_color=('white', '#005800'), size=(16, None),
                          file_types=(("text files", ".csv"), ("all files", "*.*"),))],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.Button('Show accuracy', font='Arial', size=(16, None), button_color=('white', '#005800'), )],
           [sg.Text(' ' * 30, size=(None, 1))],
           [sg.FileBrowse('Show classification', font='Arial', size=(16, None), button_color=('white', '#005800'), )]]
column3 = [[sg.Multiline('This is the log', size=(45, 20), key='-OUTPUT-')],
           ]
column4 = [[sg.Button('Reset', font='Arial', size=(16, None), button_color=('white', '#005800'))],
           ]
column5 = [[sg.Button('Quit', font='Arial', size=(16, None), button_color=('white', '#005800'))],

           ]

layout = [[sg.Column(column1, justification='center')], [sg.Column(column2, justification='l'), sg.Column(column3)],
          [sg.Text(' ' * 10), sg.Column(column4, element_justification='left'), sg.Column(column5)],

          ]

# Display the window and get values

window = sg.Window('Decision Tree', layout, text_justification='left')

while True:
    event, values = window.read()
    print(event, values)
    print(values)
    if event in (sg.WIN_CLOSED, 'Quit'):
        break
    elif event is 'Write a review':
        Review_text = sg.popup_get_text('Enter your review')
        print(Review_text)
        old_text = list(values.values())[3]
        new_text = old_text + 'User entered a review : "' + Review_text + '"'
        window['-OUTPUT-'].update(new_text)
    elif event is 'Reset':
        print("Reset Triggered , do something !")
    elif event is 'Show accuracy':
        Train_path = list(values.values())[0]
        print("Train path : " + Train_path)
        Test_path = list(values.values())[1]
        print("Test_path :" + Test_path)
        #call the decision tree function

window.close()
