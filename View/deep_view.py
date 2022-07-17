import gc
import tkinter as tk
from tkinter import Canvas, Frame, StringVar, Text, Toplevel, ttk
from tkinter.constants import BOTH, HORIZONTAL, LEFT, RIGHT, VERTICAL
import Controller.deep_findings as deep
from Model.setup_data import make_text
class deepView:
    def __init__(self) -> None:
        pass
    
    

    def create_CNN_frame(container, model, feature_type, classification_type):

            frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            
            
            ttk.Label(frame, text='Convolutional Neural Network:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
            if classification_type == 0:
                ttk.Label(frame, text='Classification : Binary-Class Classification').grid(column=0, row= 2, sticky=tk.W)
            else:
                ttk.Label(frame, text='Classification : Multi-class Classification').grid(column=0, row=2, sticky=tk.W)

            if feature_type == 0:
                ttk.Label(frame, text='Feature Set : All Features (34)').grid(column=0, row= 3, sticky=tk.W)
            elif feature_type == 1:
                ttk.Label(frame, text='Feature Set : Without IP Features (20)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 2:
                ttk.Label(frame, text='Feature Set : Top 7 features with IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 3:
                ttk.Label(frame, text='Feature Set : Top 6 features without IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 4:
                ttk.Label(frame, text='Feature Set : Top 11 features with IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
            elif feature_type == 5:
                ttk.Label(frame, text='Feature Set : Top 9 features without IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
                

            ttk.Label(frame, text='CNN  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

            display_text = make_text(model)

            frame_box = Frame(frame)
            text_box = Text(
            frame_box,
            height=25,
            width=47,
            wrap='word'
        )   
            text_box.insert('end', display_text)
            text_box.pack(side=LEFT,expand=True)
            text_box.configure(state='disabled')

            sb = ttk.Scrollbar(frame_box)
            sb.pack(side=RIGHT, fill=BOTH)

            text_box.config(yscrollcommand=sb.set)
            sb.config(command=text_box.yview)

            frame_box.grid(column=0, row=5, sticky=tk.W)
            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)

            return frame


    def create_DNN_frame(container, model, feature_type, classification_type):

            frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )

            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            
            
            ttk.Label(frame, text='Deep Neural Network:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
            if classification_type == 0:
                ttk.Label(frame, text='Classification : Binary-Class Classification').grid(column=0, row= 2, sticky=tk.W)
            else:
                ttk.Label(frame, text='Classification : Multi-class Classification').grid(column=0, row=2, sticky=tk.W)

            if feature_type == 0:
                ttk.Label(frame, text='Feature Set : All Features (34)').grid(column=0, row= 3, sticky=tk.W)
            elif feature_type == 1:
                ttk.Label(frame, text='Feature Set : Without IP Features (20)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 2:
                ttk.Label(frame, text='Feature Set : Top 7 features with IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 3:
                ttk.Label(frame, text='Feature Set : Top 6 features without IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 4:
                ttk.Label(frame, text='Feature Set : Top 11 features with IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
            elif feature_type == 5:
                ttk.Label(frame, text='Feature Set : Top 9 features without IP Features (IG)').grid(column=0, row=3, sticky=tk.W)

            ttk.Label(frame, text='DNN  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

            display_text = make_text(model)
            

            frame_box = Frame(frame)
            text_box = Text(
            frame_box,
            height=25,
            width=45,
            wrap='word'
        )   
            text_box.insert('end', display_text)
            text_box.pack(side=LEFT,expand=True)
            text_box.configure(state='disabled')


            sb = ttk.Scrollbar(frame_box)
            sb.pack(side=RIGHT, fill=BOTH)

            text_box.config(yscrollcommand=sb.set)
            sb.config(command=text_box.yview)

            frame_box.grid(column=0, row=5, sticky=tk.W)
            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)

            return frame

    def create_RNN_frame(container, model, feature_type, classification_type):

            frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            
            
            ttk.Label(frame, text='Recurrent Neural Network:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
            if classification_type == 0:
                ttk.Label(frame, text='Classification : Binary-Class Classification').grid(column=0, row= 2, sticky=tk.W)
            else:
                ttk.Label(frame, text='Classification : Multi-class Classification').grid(column=0, row=2, sticky=tk.W)

            if feature_type == 0:
                ttk.Label(frame, text='Feature Set : All Features (34)').grid(column=0, row= 3, sticky=tk.W)
            elif feature_type == 1:
                ttk.Label(frame, text='Feature Set : Without IP Features (20)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 2:
                ttk.Label(frame, text='Feature Set : Top 7 features with IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 3:
                ttk.Label(frame, text='Feature Set : Top 6 features without IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 4:
                ttk.Label(frame, text='Feature Set : Top 11 features with IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
            elif feature_type == 5:
                ttk.Label(frame, text='Feature Set : Top 9 features without IP Features (IG)').grid(column=0, row=3, sticky=tk.W)

            ttk.Label(frame, text='RNN  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

            display_text = make_text(model)

            
            frame_box = Frame(frame)
            text_box = Text(
            frame_box,
            height=25,
            width=45,
            wrap='word'
        )   
            text_box.insert('end', display_text)
            text_box.pack(side=LEFT,expand=True)
            text_box.configure(state='disabled')


            sb = ttk.Scrollbar(frame_box)
            sb.pack(side=RIGHT, fill=BOTH)

            text_box.config(yscrollcommand=sb.set)
            sb.config(command=text_box.yview)

            frame_box.grid(column=0, row=5, sticky=tk.W)
            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)

            return frame

    def create_LSTM_frame(container, model, feature_type, classification_type):

            frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            
            
            ttk.Label(frame, text='Long Short Term Memory:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
            if classification_type == 0:
                ttk.Label(frame, text='Classification : Binary-Class Classification').grid(column=0, row= 2, sticky=tk.W)
            else:
                ttk.Label(frame, text='Classification : Multi-class Classification').grid(column=0, row=2, sticky=tk.W)

            if feature_type == 0:
                ttk.Label(frame, text='Feature Set : All Features (34)').grid(column=0, row= 3, sticky=tk.W)
            elif feature_type == 1:
                ttk.Label(frame, text='Feature Set : Without IP Features (20)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 2:
                ttk.Label(frame, text='Feature Set : Top 7 features with IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 3:
                ttk.Label(frame, text='Feature Set : Top 6 features without IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 4:
                ttk.Label(frame, text='Feature Set : Top 11 features with IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
            elif feature_type == 5:
                ttk.Label(frame, text='Feature Set : Top 9 features without IP Features (IG)').grid(column=0, row=3, sticky=tk.W)

            ttk.Label(frame, text='LSTM Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)


            display_text = make_text(model)

            
            frame_box = Frame(frame)
            text_box = Text(
            frame_box,
            height=24,
            width=45,
            wrap='word'
        )   
            text_box.insert('end', display_text)
            text_box.pack(side=LEFT,expand=True)
            text_box.configure(state='disabled')


            sb = ttk.Scrollbar(frame_box)
            sb.pack(side=RIGHT, fill=BOTH)

            text_box.config(yscrollcommand=sb.set)
            
            sb.config(command=text_box.yview)

            frame_box.grid(column=0, row=5, sticky=tk.W)
            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)

            return frame

    def create_MLP_frame(container, model, feature_type, classification_type):

            frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            
            
            ttk.Label(frame, text='Multi-Layer Perceptron:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
            if classification_type == 0:
                ttk.Label(frame, text='Classification : Binary-Class Classification').grid(column=0, row= 2, sticky=tk.W)
            else:
                ttk.Label(frame, text='Classification : Multi-class Classification').grid(column=0, row=2, sticky=tk.W)

            if feature_type == 0:
                ttk.Label(frame, text='Feature Set : All Features (34)').grid(column=0, row= 3, sticky=tk.W)
            elif feature_type == 1:
                ttk.Label(frame, text='Feature Set : Without IP Features (20)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 2:
                ttk.Label(frame, text='Feature Set : Top 7 features with IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 3:
                ttk.Label(frame, text='Feature Set : Top 6 features without IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 4:
                ttk.Label(frame, text='Feature Set : Top 11 features with IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
            elif feature_type == 5:
                ttk.Label(frame, text='Feature Set : Top 9 features without IP Features (IG)').grid(column=0, row=3, sticky=tk.W)


            ttk.Label(frame, text='MLP  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)

            count = 5
            model = deep.getMLP(feature_type, classification_type)
            for i in model.get_params():
                ttk.Label(frame, text= i + ':   '+  str(model.get_params()[i])).grid(column=0, row=count, sticky=tk.W)
                count = count +1

            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)

            return frame

    def create_DBN_frame(container, model, feature_type, classification_type):

            frame = ttk.Frame(container, height= 300, width= container.winfo_width()/2 )
            
            # grid layout for the input frame
            frame.columnconfigure(0, weight=10)
            frame.columnconfigure(0, weight=10)
            
            
            ttk.Label(frame, text='Deep Belief Network:', font=( 10)).grid(column=0, row=0, sticky=tk.W)
            if classification_type == 0:
                ttk.Label(frame, text='Classification : Binary-Class Classification').grid(column=0, row= 2, sticky=tk.W)
            else:
                ttk.Label(frame, text='Classification : Multi-class Classification').grid(column=0, row=2, sticky=tk.W)

    
            if feature_type == 0:
                ttk.Label(frame, text='Feature Set : All Features (34)').grid(column=0, row= 3, sticky=tk.W)
            elif feature_type == 1:
                ttk.Label(frame, text='Feature Set : Without IP Features (20)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 2:
                ttk.Label(frame, text='Feature Set : Top 7 features with IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 3:
                ttk.Label(frame, text='Feature Set : Top 6 features without IP Features (CFS)').grid(column=0, row=3, sticky=tk.W)
            
            elif feature_type == 4:
                ttk.Label(frame, text='Feature Set : Top 11 features with IP Features (IG)').grid(column=0, row=3, sticky=tk.W)
            elif feature_type == 5:
                ttk.Label(frame, text='Feature Set : Top 9 features without IP Features (IG)').grid(column=0, row=3, sticky=tk.W)

            ttk.Label(frame, text='DBN  Model Parameters:', font=( 10)).grid(column=0, row=4, sticky=tk.W)
            count = 5
            for i in model.get_params():
                ttk.Label(frame, text= i + ':   '+  str(model.get_params()[i])).grid(column=0, row=count, sticky=tk.W)
                count = count +1

            for widget in frame.winfo_children():
                widget.grid(padx=5, pady=5)

            return frame

   