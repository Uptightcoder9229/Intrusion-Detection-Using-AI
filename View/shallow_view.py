import tkinter as tk
from tkinter import Canvas, Frame, StringVar, Text, Toplevel, ttk

class shallowView:
    def __init__(self) -> None:
        pass

    def create_KNN_frame(container, model, feature_type, classification_type):
        frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
        variable = StringVar(frame)
        variable_1 = StringVar(frame)
        
        # grid layout for the input frame
        frame.columnconfigure(0, weight=10)
        frame.columnconfigure(0, weight=10)
        
        
        ttk.Label(frame, text='K-Nearest Neighbors  Model:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
        if classification_type == 0:
            ttk.Label(frame, text='Classification :').grid(column=0, row= 2, sticky=tk.W)
            ttk.Label(frame, text='Binary-Class Classification').grid(column=1, row=2, sticky=tk.W)
        else:
            ttk.Label(frame, text='Classification :').grid(column=0, row=2, sticky=tk.W)
            ttk.Label(frame, text='Multi-class Classification').grid(column=1, row=2, sticky=tk.W)

        if feature_type == 0:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row= 3, sticky=tk.W)
            ttk.Label(frame, text='All Features (34)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 1:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Without IP Features (20)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 2:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 7 features with IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 3:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 6 features without IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 4:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 11 features with IP Features (IG)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 5:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 9 features without IP Features (IG)').grid(column=1, row=3, sticky=tk.W)

        ttk.Label(frame, text='KNN  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

        count = 5

        for i in model.get_params():
            ttk.Label(frame, text= i + ': ').grid(column=0, row=count, sticky=tk.W)
            ttk.Label(frame, width=30, text = str(model.get_params()[i]) ).grid(column=1, row=count, sticky=tk.W)
            count = count +1

        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)

        return frame

    def create_SVM_frame(container, model, feature_type, classification_type):

        frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
        variable = StringVar(frame)
        variable_1 = StringVar(frame)
        
        # grid layout for the input frame
        frame.columnconfigure(0, weight=10)
        frame.columnconfigure(0, weight=10)
        
        
        ttk.Label(frame, text='SVM  Model:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
        if classification_type == 0:
            ttk.Label(frame, text='Classification :').grid(column=0, row= 2, sticky=tk.W)
            ttk.Label(frame, text='Binary-Class Classification').grid(column=1, row=2, sticky=tk.W)
        else:
            ttk.Label(frame, text='Classification :').grid(column=0, row=2, sticky=tk.W)
            ttk.Label(frame, text='Multi-class Classification').grid(column=1, row=2, sticky=tk.W)

        if feature_type == 0:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row= 3, sticky=tk.W)
            ttk.Label(frame, text='All Features (34)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 1:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Without IP Features (20)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 2:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 7 features with IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 3:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 6 features without IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 4:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 11 features with IP Features (IG)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 5:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 9 features without IP Features (IG)').grid(column=1, row=3, sticky=tk.W)

        ttk.Label(frame, text='SVM  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

        count = 5

        for i in model.get_params():
            ttk.Label(frame, text= i + ': ').grid(column=0, row=count, sticky=tk.W)
            ttk.Label(frame, width=30, text = str(model.get_params()[i]) ).grid(column=1, row=count, sticky=tk.W)
            count = count +1

        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)        
        return frame

    def create_DT_frame(container, model, feature_type, classification_type):
        frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
        variable = StringVar(frame)
        variable_1 = StringVar(frame)
        
        # grid layout for the input frame
        frame.columnconfigure(0, weight=10)
        frame.columnconfigure(0, weight=10)
        
        
        ttk.Label(frame, text='Decision Tree Model:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
        if classification_type == 0:
            ttk.Label(frame, text='Classification :').grid(column=0, row= 2, sticky=tk.W)
            ttk.Label(frame, text='Binary-Class Classification').grid(column=1, row=2, sticky=tk.W)
        else:
            ttk.Label(frame, text='Classification :').grid(column=0, row=2, sticky=tk.W)
            ttk.Label(frame, text='Multi-class Classification').grid(column=1, row=2, sticky=tk.W)

        if feature_type == 0:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row= 3, sticky=tk.W)
            ttk.Label(frame, text='All Features (34)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 1:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Without IP Features (20)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 2:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 7 features with IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 3:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 6 features without IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 4:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 11 features with IP Features (IG)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 5:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 9 features without IP Features (IG)').grid(column=1, row=3, sticky=tk.W)
        ttk.Label(frame, text='DT Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

        count = 5

        for i in model.get_params():
            ttk.Label(frame, text= i + ': ').grid(column=0, row=count, sticky=tk.W)
            ttk.Label(frame, width=30, text = str(model.get_params()[i]) ).grid(column=1, row=count, sticky=tk.W)
            count = count +1

        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)        
        return frame

    def create_RF_frame(container, model, feature_type, classification_type):
        frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
        variable = StringVar(frame)
        variable_1 = StringVar(frame)
        
        # grid layout for the input frame
        frame.columnconfigure(0, weight=10)
        frame.columnconfigure(0, weight=10)
        
        
        ttk.Label(frame, text='Random Forest  Model:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
        if classification_type == 0:
            ttk.Label(frame, text='Classification :').grid(column=0, row= 2, sticky=tk.W)
            ttk.Label(frame, text='Binary-Class Classification').grid(column=1, row=2, sticky=tk.W)
        else:
            ttk.Label(frame, text='Classification :').grid(column=0, row=2, sticky=tk.W)
            ttk.Label(frame, text='Multi-class Classification').grid(column=1, row=2, sticky=tk.W)

        if feature_type == 0:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row= 3, sticky=tk.W)
            ttk.Label(frame, text='All Features (34)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 1:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Without IP Features (20)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 2:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 7 features with IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 3:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 6 features without IP Features (CFS)').grid(column=1, row=3, sticky=tk.W)
        
        elif feature_type == 4:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 11 features with IP Features (IG)').grid(column=1, row=3, sticky=tk.W)
        elif feature_type == 5:
            ttk.Label(frame, text='Feature Set :').grid(column=0, row=3, sticky=tk.W)
            ttk.Label(frame, text='Top 9 features without IP Features (IG)').grid(column=1, row=3, sticky=tk.W)

        ttk.Label(frame, text='RF Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

        count = 5

        for i in model.get_params():
            ttk.Label(frame, text= i + ': ').grid(column=0, row=count, sticky=tk.W)
            ttk.Label(frame, width=30, text = str(model.get_params()[i]) ).grid(column=1, row=count, sticky=tk.W)
            count = count +1
        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)        
        return frame