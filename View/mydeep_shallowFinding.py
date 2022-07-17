import gc
from logging import root
import tkinter as tk
from tkinter import Canvas, Frame, StringVar, Text, Toplevel, ttk
from tkinter.constants import BOTH, HORIZONTAL, LEFT, RIGHT, VERTICAL
import tkinter.messagebox
from typing import Container
from Model.setup_data import feature_extractor_set, flatten, get_feature_extractor, preprocess_data, test_deep_model, test_single_model, test_split, plot_classification_report,plot_confusion_matrix, plot_deep_model, make_4D
import threading
import Controller.deep_findings as deep
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from tensorflow.keras.utils import plot_model
import View.deep_view as deepFrame
from View.figure_view import create_plot_window
from Controller.deep_shallowfindings import *

class mydeep_shallowFindings:
    old_rate = None
    X = None
    Y = None
    pro = None
    X_2 = None
    Y_2 = None 
    pro = None
    dataset_choice = None
    secondModel = None
    current_model = None
    def __init__(self, data, modeltype, feature_type, classification_type):
        self.old_rate = None
        self.X = None
        self.Y = None
        self.X_2 = None
        self.Y_2 = None 
        self.pro = None
        self.dataset_choice = None
        self.secondModel = None
        self.current_model = None
        self.create_main_window(data, modeltype, feature_type, classification_type)

    
    def new_train_performance(self, data, root, second_model, modeltype, feature_type, classification_type, preprocess_status, preprocess_canvas, split_rate, variableDataset, model_DL, old_split_rate = '20'):
        split_rate = split_rate.strip()
        old_split_rate = self.old_rate
        
        if split_rate.strip().isnumeric() == False:
            tkinter.messagebox.showwarning(title='Error', message='Test Size Split is not a Number')
        elif preprocess_canvas.itemcget(preprocess_status, 'fill') == 'red':
            tkinter.messagebox.showwarning(title='Error', message='The dataset is still being preprocessed')
        elif (float(split_rate) > 100) & (float(split_rate) >=0):
                tkinter.messagebox.showwarning(title='Error', message='The Test Split rate is not in range 1 - 100')

        

        elif (old_split_rate == split_rate) & (second_model == self.secondModel) & (self.dataset_choice == variableDataset) & (self.current_model == model_DL):
            if (old_split_rate == split_rate) & (self.dataset_choice == variableDataset):
                tkinter.messagebox.showwarning(title='Error', message='The Test Split rate or Dataset has not been changed')
            elif second_model == self.secondModel:
                tkinter.messagebox.showwarning(title='Error', message='The Second Model has not changed')
        
        else:    
            for widget in root.winfo_children():
                widget.place_forget()
                widget.destroy()
            ttk.Label(root, text = "Preprocessing Status --->").place(relx = 0.875, rely = 0, anchor='n')
            preprocess_canvas =  Canvas(root, width=15, height=15)
            preprocess_status = preprocess_canvas.create_oval(15, 15, 5, 5)
            preprocess_canvas.itemconfig(preprocess_status, fill="red")
            preprocess_canvas.place(relx = .95, rely = 0, anchor='n')
            ttk.Label(root, text= 'TestSet Split: ').place(relx = 0.775, rely = 0.05, anchor='n')
            split = ttk.Entry(root, width=10)
            split.insert(0, split_rate)
            split.place(relx = .85, rely = 0.05, anchor='n')

            
            variable_DLmodel = StringVar(root)
            availableDLModel = ['CNN', 'RNN', 'LSTM']
            variable_DLmodel.set(model_DL) # default value
            ttk.Label(root, text='DL-Algorithm:').place(relx = .9, rely = 0.1, anchor='n')
            DL_model = tk.OptionMenu(root, variable_DLmodel, *availableDLModel)
            DL_model.place(relx = 0.97, rely = 0.1, anchor='n')

            variable_model = StringVar(root)
            availableModel = ['KNN', 'SVM', 'RF', 'DT']
            variable_model.set(second_model) # default value
            ttk.Label(root, text='SL-Algorithm:').place(relx = .9, rely = 0.15, anchor='n')
            second_model = tk.OptionMenu(root, variable_model, *availableModel)
            second_model.place(relx = 0.97, rely = 0.15, anchor='n')

            variable_dataset = StringVar(root)
            availabledataset = ['CIDDS-001', 'CIDDS-002']
            variable_dataset.set(variableDataset) # default value
            ttk.Label(root, text='Dataset:').place(relx = .745, rely = 0.1, anchor='n')
            dataset = tk.OptionMenu(root, variable_dataset, *availabledataset)
            dataset.place(relx = .82, rely = 0.1, anchor='n')

            tk.Button(root, text='Test Model', width = 15, command= (lambda: self.new_train_performance
            (data, root, variable_model.get(), modeltype, feature_type, classification_type, preprocess_status, preprocess_canvas, split.get(), variable_dataset.get(), variable_DLmodel.get()))).place(relx = 0.94, rely = 0.045, anchor='n')
            make4d = False
            self.current_model = variable_DLmodel.get()
            if variable_DLmodel.get() == 'CNN':
                model = getCNN_shallow(feature_type, classification_type)
                open = deepFrame.deepView.create_CNN_frame( root, model['main'], feature_type, classification_type)
                model['main'] = get_feature_extractor(model['main'], 'flatten')
                open.place(relx = 0.21, rely = 0.03, anchor='n')
            elif variable_DLmodel.get() == 'RNN':
                    model = getRNN_shallow(feature_type, classification_type)
                    open = deepFrame.deepView.create_RNN_frame( root, model['main'], feature_type, classification_type)
                    model['main'] = get_feature_extractor(model['main'], 'flatten')
                    open.place(relx = 0.21, rely = 0.03, anchor='n')
            elif variable_DLmodel.get() == 'LSTM':
                    model = getLSTM_shallow(feature_type, classification_type)
                    open = deepFrame.deepView.create_LSTM_frame( root, model['main'],  feature_type, classification_type)
                    model['main'] = get_feature_extractor(model['main'], 'flatten')
                    open.place(relx = 0.21, rely = 0.03, anchor='n')
            
            frame = ttk.Frame(root, height= 300, width= root.winfo_width()/4 )
            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            

            ttk.Label(frame, text= variable_model.get() + '  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

            count = 5

            for i in model[variable_model.get()].get_params():
                ttk.Label(frame, text= i + ': ').grid(column=0, row=count, sticky=tk.W)
                ttk.Label(frame, width=30, text = str(model[variable_model.get()].get_params()[i]) ).grid(column=1, row=count, sticky=tk.W)
                count = count +1

            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)
            
            frame.place(relx = 0.525, rely = 0.12, anchor='n')
            #variable_model.trace([variable_model.get(),model[variable_model.get()], root], self.option_changed)
            

            self.pro = threading.Thread(target= self.load_preprocess, 
                args=(root, model, variable_model.get(), feature_type, classification_type, preprocess_status, preprocess_canvas, variable_dataset.get(), split_rate, make4d))
            self.pro.setDaemon(True)
            self.pro.start()
            

    def performance_frame(self, container, model, feature_type, classification_type, preprocess_status, preprocess_canvas, X_test, Y_test, axes):
        frame = ttk.Frame(container, height= 300, width= container.winfo_width()/6 )
        
        ttk.Label(frame, text= "Performance Of Model:", font=(10)).grid(column=0, row=0, sticky=tk.W)
        
        # grid layout for the input frame
        frame.columnconfigure(0, weight=10)
        frame.columnconfigure(0, weight=10)

        count = 2
        performance_parameters , roc_plot, classification_matrix, clf_report = test_single_model(model, X_test, Y_test, classification_type, axes)
        for i in performance_parameters.keys():
            ttk.Label(frame, text= i + ': ').grid(column=1, row=count, sticky=tk.W)
            ttk.Label(frame, width=30, text = str(performance_parameters[i]) ).grid(column=2, row=count, sticky=tk.W)
            count = count +1
        preprocess_canvas.itemconfig(preprocess_status, fill="green")
        for widget in frame.winfo_children():
            widget.grid(padx=2, pady=2)        
        return frame, roc_plot, classification_matrix, clf_report

    def load_preprocess(self, root, model, second_model, feature_type, classification_type, preprocess_status, preprocess_canvas, variable_dataset = 'CIDDS-001', split = '20', make4D = False):
        self.old_rate = split
        self.secondModel = second_model
        self.dataset_choice = variable_dataset
        if split != '100':
            if variable_dataset == 'CIDDS-001':
                X_test, Y_test = test_split(self.X,self.Y, float(split))
            else:
                X_test, Y_test = test_split(self.X_2,self.Y_2, float(split))
        else:
            if variable_dataset == 'CIDDS-001':
                X_test, Y_test = self.X, self.Y
            else:
                X_test, Y_test = self.X_2, self.Y_2
        
        X_test = feature_extractor_set(model['main'], X_test)
        figure = plt.Figure(figsize= (16, 16))
        gs = figure.add_gridspec(21, 21)
        axe1 = figure.add_subplot(gs[0:6, 1:7])
        axe2 = figure.add_subplot(gs[12:, 1:7])
        axe3 = figure.add_subplot(gs[0: , 9:16])
        axe4 = figure.add_subplot(gs[0:, 17:])
            
        if classification_type == 0:
            labels = ['Normal', 'Attack']
        else:
            labels =  ['Normal', 'BruteForce', 'PingScan', 'PortScan']
        performance, axe1, classification_matrix, clf_report = self.performance_frame(root, model[second_model], feature_type, classification_type, preprocess_status, preprocess_canvas, X_test, Y_test, axe1)
        performance.place(relx = 1, rely = 0.6, anchor='e')
        axe3 = plot_classification_report(clf_report, axe3)
        axe2 = plot_confusion_matrix(classification_matrix, labels, axe2)
        axe4 = plot_deep_model(model['main'], axe4)
        #axe1.axis('scaled')
        axe2.axis('scaled')
        axe3.axis('scaled')
        #axe4.axis('auto')
        ttk.Label(root, text= 'Total Number of Instances in test size: '+ str(len(Y_test)) ).place(relx = 0.7, rely = 0.15, anchor='n')
        figure.tight_layout()
        create_plot_window(figure)
        print('DOne')

    def start_preprocess(self, data, feature_type, classification_type, root, model, second_model, preprocess_status, preprocess_canvas, make4d = False): 
        self.X, self.Y = preprocess_data(data['001'].copy(), feature_type, classification_type)
        self.X_2, self.Y_2 = preprocess_data(data['002'].copy(), feature_type, classification_type)
        self.load_preprocess(root, model, second_model, feature_type, classification_type, preprocess_status, preprocess_canvas,'CIDDS-001','20', make4d)
        

    
    

    def create_main_window(self, data, modeltype, feature_type, classification_type ):
        gc.collect()
        # root window
        root = tk.Tk()
        root.title(modeltype + ' Model Findings')
        root.geometry('1300x600')
        root.resizable(0, 0)
        # windows only (remove the minimize/maximize button)
        #root.attributes('-toolwindow', True)
        ttk.Label(root, text = "Preprocessing Status --->").place(relx = 0.875, rely = 0, anchor='n')
        preprocess_canvas =  Canvas(root, width=15, height=15)
        preprocess_status = preprocess_canvas.create_oval(15, 15, 5, 5)
        preprocess_canvas.itemconfig(preprocess_status, fill="red")
        preprocess_canvas.place(relx = .95, rely = 0, anchor='n')
        ttk.Label(root, text= 'TestSet Split: ').place(relx = 0.775, rely = 0.05, anchor='n')
        split = ttk.Entry(root, width=10)
        split.insert(0, '20')
        split.place(relx = .85, rely = 0.05, anchor='n')

        variable_DLmodel = StringVar(root)
        availableDLModel = ['CNN', 'RNN', 'LSTM']
        variable_DLmodel.set(availableDLModel[0]) # default value
        ttk.Label(root, text='DL-Algorithm:').place(relx = .9, rely = 0.1, anchor='n')
        DL_model = tk.OptionMenu(root, variable_DLmodel, *availableDLModel)
        DL_model.place(relx = 0.97, rely = 0.1, anchor='n')

        variable_model = StringVar(root)
        availableModel = ['KNN', 'SVM', 'RF', 'DT']
        variable_model.set(availableModel[0]) # default value
        ttk.Label(root, text='SL-Algorithm:').place(relx = .9, rely = 0.15, anchor='n')
        second_model = tk.OptionMenu(root, variable_model, *availableModel)
        second_model.place(relx = 0.97, rely = 0.15, anchor='n')


        variable_dataset = StringVar(root)
        availabledataset = ['CIDDS-001', 'CIDDS-002']
        variable_dataset.set(availabledataset[0]) # default value
        ttk.Label(root, text='Dataset:').place(relx = .76, rely = 0.1, anchor='n')
        dataset = tk.OptionMenu(root, variable_dataset, *availabledataset)
        dataset.place(relx = .82, rely = 0.1, anchor='n')

        tk.Button(root, text='Test Model', width = 15, command= (lambda: self.new_train_performance
            (data, root, variable_model.get(), modeltype, feature_type, classification_type, 
            preprocess_status, preprocess_canvas, split.get(), variable_dataset.get(), variable_DLmodel.get()))).place(relx = 0.94, rely = 0.045, anchor='n')
        make4d = False
        self.current_model = variable_DLmodel.get()
        if variable_DLmodel.get() == 'CNN':
                model = getCNN_shallow(feature_type, classification_type)
                open = deepFrame.deepView.create_CNN_frame( root, model['main'], feature_type, classification_type)
                model['main'] = get_feature_extractor(model['main'], 'flatten')
                open.place(relx = 0.21, rely = 0.03, anchor='n')
        elif variable_DLmodel.get() == 'RNN':
                model = getRNN_shallow(feature_type, classification_type)
                open = deepFrame.deepView.create_RNN_frame( root, model['main'], feature_type, classification_type)
                model['main'] = get_feature_extractor(model['main'], 'flatten')
                open.place(relx = 0.21, rely = 0.03, anchor='n')
        elif variable_DLmodel.get() == 'LSTM':
                model = getLSTM_shallow(feature_type, classification_type)
                open = deepFrame.deepView.create_LSTM_frame( root, model['main'],  feature_type, classification_type)
                model['main'] = get_feature_extractor(model['main'], 'flatten')
                open.place(relx = 0.21, rely = 0.03, anchor='n')
        
        frame = ttk.Frame(root, height= 300, width= root.winfo_width()/4 )
        
        # grid layout for the input frame
        frame.columnconfigure(0, weight=10)
        frame.columnconfigure(0, weight=10)
        

        ttk.Label(frame, text= variable_model.get() + '  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

        count = 5

        for i in model[variable_model.get()].get_params():
            ttk.Label(frame, text= i + ': ').grid(column=0, row=count, sticky=tk.W)
            ttk.Label(frame, width=30, text = str(model[variable_model.get()].get_params()[i]) ).grid(column=1, row=count, sticky=tk.W)
            count = count +1

        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)
        
        frame.place(relx = 0.525, rely = 0.12, anchor='n')
        self.pro = threading.Thread(target= self.start_preprocess, 
            args = (data, feature_type, classification_type, root, model, variable_model.get(), preprocess_status, preprocess_canvas,  make4d)
    )
        self.pro.setDaemon(True)
        self.pro.start()

    

        #Frame for performace evaluation
        
        # layout on the root window
        

        root.mainloop()
        return True