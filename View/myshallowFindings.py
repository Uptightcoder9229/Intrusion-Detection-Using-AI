import gc
import threading
import tkinter as tk
from tkinter import Canvas, StringVar, Toplevel, XView, YView, ttk
import tkinter.messagebox
from Model.setup_data import flatten, preprocess_data, test_single_model, test_split, plot_confusion_matrix,plot_classification_report
import View.shallow_view as shallowFrame
import matplotlib.pyplot as plt
from Controller.shallow_findings  import *


from View.figure_view import create_plot_window
#0: Binary 1: Multiclass
#0: ALL 1: 20S
class myshallowFindings:
    old_rate = None
    X = None
    Y = None 
    X_2 = None
    Y_2 = None 
    pro = None
    dataset_choice = None
    current_model = None
    def __init__(self, data, modeltype, feature_type, classification_type):
        self.old_rate = None
        self.X = None
        self.Y = None
        self.X_2 = None
        self.Y_2 = None 
        self.pro = None
        self.dataset_choice = None
        self.current_model = None
        self.create_main_window(data, modeltype, feature_type, classification_type)

    def new_train_performance(self, data, root, modeltype, feature_type, classification_type, preprocess_status, preprocess_canvas, split_rate, variableDataset ,old_split_rate = '20'):
        split_rate = split_rate.strip()
        old_split_rate = self.old_rate
        if split_rate.strip().isnumeric() == False:
            tkinter.messagebox.showwarning(title='Error', message='Test Size Split is not a Number')
        elif preprocess_canvas.itemcget(preprocess_status, 'fill') == 'red':
            tkinter.messagebox.showwarning(title='Error', message='The dataset is still being preprocessed')
        elif (float(split_rate) > 100) & (float(split_rate) >=0):
            tkinter.messagebox.showwarning(title='Error', message='The Test Split rate is not in range 1 - 100')

        elif (old_split_rate == split_rate) & (self.dataset_choice == variableDataset) & (self.current_model == modeltype):
            tkinter.messagebox.showwarning(title='Error', message='The Test Split rate or Dataset has not been changed')
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

            variable_model = StringVar(root)
            availableModel = ['KNN', 'SVM', 'RF', 'DT']
            variable_model.set(modeltype) # default value
            ttk.Label(root, text='SL-Algorithm:').place(relx = .9, rely = 0.1, anchor='n')
            second_model = tk.OptionMenu(root, variable_model, *availableModel)
            second_model.place(relx = 0.97, rely = 0.1, anchor='n')

            variable_dataset = StringVar(root)
            availabledataset = ['CIDDS-001', 'CIDDS-002']
            variable_dataset.set(variableDataset) # default value
            ttk.Label(root, text='Dataset:').place(relx = .765, rely = 0.1, anchor='n')
            dataset = tk.OptionMenu(root, variable_dataset, *availabledataset)
            dataset.place(relx = .82, rely = 0.1, anchor='n')

            tk.Button(root, text='Test Model', width = 15, command= (lambda: self.new_train_performance
            (data, root, variable_model.get(), feature_type, classification_type, preprocess_status, preprocess_canvas, split.get(), variable_dataset.get()))).place(relx = 0.94, rely = 0.045, anchor='n')
            self.current_model = modeltype
            if variable_model.get() == 'KNN':
                model = getKNN(feature_type, classification_type)
                open = shallowFrame.shallowView.create_KNN_frame( root, model, feature_type, classification_type)
                open.place(relx = 0.2, rely = 0.05, anchor='n')
            elif variable_model.get() == 'SVM':
                model = getSVM(feature_type, classification_type)
                open = shallowFrame.shallowView.create_SVM_frame( root, model, feature_type, classification_type)
                open.place(relx = 0.2, rely = 0.05, anchor='n')
            elif variable_model.get() == 'DT':
                model = getDT(feature_type, classification_type)
                open= shallowFrame.shallowView.create_DT_frame( root, model, feature_type, classification_type)
                open.place(relx = 0.2, rely = 0.05, anchor='n')
            elif variable_model.get() == 'RF':
                model = getRF(feature_type, classification_type)
                open = shallowFrame.shallowView.create_RF_frame( root, model, feature_type, classification_type)
                open.place(relx = 0.2, rely = 0.05, anchor='n')

            self.pro = threading.Thread(target= self.load_preprocess, 
            args=(root, model, feature_type, classification_type, preprocess_status, preprocess_canvas, variable_dataset.get(), split_rate))
            self.pro.setDaemon(True)
            self.pro.start()
            

    def performance_frame(self, container, model, feature_type, classification_type, preprocess_status, preprocess_canvas, X_test, Y_test, axes):
        frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )

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

    def load_preprocess(self, root, model, feature_type, classification_type, preprocess_status, preprocess_canvas , variable_dataset = 'CIDDS-001',split = '20'):
       
        self.old_rate = split
        self.dataset_choice = variable_dataset
        if split != '100':
            if variable_dataset == 'CIDDS-001':
                X_test, Y_test = test_split(flatten(self.X),self.Y, float(split))
            else:
                X_test, Y_test = test_split(flatten(self.X_2),self.Y_2, float(split))
        else:
            if variable_dataset == 'CIDDS-001':
                X_test, Y_test = flatten(self.X), self.Y
            else:
                X_test, Y_test = flatten(self.X_2), self.Y_2
        figure = plt.Figure(figsize= (16, 16))
        gs = figure.add_gridspec(22, 22)
        axe1 = figure.add_subplot(gs[0:, 1:7])
        axe2 = figure.add_subplot(gs[0:, 9:15])
        axe3 = figure.add_subplot(gs[0: , 17:])
        
        if classification_type == 0:
            labels = ['Normal', 'Attack']
        else:
            labels =  ['Normal', 'BruteForce', 'PingScan', 'PortScan']
        performance, axe1, classification_matrix, clf_report = self.performance_frame(root, model, feature_type, classification_type, preprocess_status, preprocess_canvas, X_test, Y_test, axe1)
        performance.place(relx = 1, rely = 0.5, anchor='e')
        axe3 = plot_classification_report(clf_report, axe3)
        axe2 = plot_confusion_matrix(classification_matrix, labels, axe2) 
        axe1.axis('scaled')
        axe2.axis('scaled')
        axe3.axis('scaled')
        ttk.Label(root, text= 'Total Number of Instances in test size: '+ str(len(Y_test)) ).place(relx = 0.7, rely = 0.15, anchor='n')
        figure.tight_layout()
        create_plot_window(figure)
        print('Done')
        
    def start_preprocess(self, data, feature_type, classification_type, root, model, preprocess_status, preprocess_canvas) :
        self.X, self.Y = preprocess_data(data['001'].copy(), feature_type, classification_type)
        self.X_2, self.Y_2 = preprocess_data(data['002'].copy(), feature_type, classification_type)
        self.load_preprocess(root, model, feature_type, classification_type, preprocess_status, preprocess_canvas)

    def create_main_window(self, data, modeltype, feature_type, classification_type ):
        gc.collect()
        # root window
        root = tk.Tk()
        root.title(modeltype + ' Model Findings')
        root.geometry('1200x700')
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

        variable_model = StringVar(root)
        availableModel = ['KNN', 'SVM', 'RF', 'DT']
        variable_model.set(availableModel[0]) # default value
        ttk.Label(root, text='SL-Algorithm:').place(relx = .9, rely = 0.1, anchor='n')
        second_model = tk.OptionMenu(root, variable_model, *availableModel)
        second_model.place(relx = 0.97, rely = 0.1, anchor='n')

        variable_dataset = StringVar(root)
        availabledataset = ['CIDDS-001', 'CIDDS-002']
        variable_dataset.set(availabledataset[0]) # default value
        ttk.Label(root, text='Dataset:').place(relx = .745, rely = 0.1, anchor='n')
        dataset = tk.OptionMenu(root, variable_dataset, *availabledataset)
        dataset.place(relx = .82, rely = 0.1, anchor='n')

        tk.Button(root, text='Test Model', width = 15, command= (lambda: self.new_train_performance
        (data, root, variable_model.get(), feature_type, classification_type, preprocess_status, preprocess_canvas, split.get(), variable_dataset.get()))).place(relx = 0.94, rely = 0.045, anchor='n')
        
        self.current_model = variable_model.get()
        if variable_model.get() == 'KNN':
            model = getKNN(feature_type, classification_type)
            open = shallowFrame.shallowView.create_KNN_frame( root, model, feature_type, classification_type)
            open.place(relx = 0.2, rely = 0.05, anchor='n')
        elif variable_model.get() == 'SVM':
            model = getSVM(feature_type, classification_type)
            open = shallowFrame.shallowView.create_SVM_frame( root, model, feature_type, classification_type)
            open.place(relx = 0.2, rely = 0.05, anchor='n')
        elif variable_model.get() == 'DT':
            model = getDT(feature_type, classification_type)
            open= shallowFrame.shallowView.create_DT_frame( root, model, feature_type, classification_type)
            open.place(relx = 0.2, rely = 0.05, anchor='n')
        elif variable_model.get() == 'RF':
            model = getRF(feature_type, classification_type)
            open = shallowFrame.shallowView.create_RF_frame( root, model, feature_type, classification_type)
            open.place(relx = 0.2, rely = 0.05, anchor='n')

        self.pro = threading.Thread(target= self.start_preprocess, 
            args = (data, feature_type, classification_type, root, model, preprocess_status, preprocess_canvas)
    )   
        self.pro.setDaemon(True)
        self.pro.start()

    
        #Frame for performace evaluation
        
        # layout on the root window
        

        root.mainloop()
        return True


