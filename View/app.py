from os import error
import tkinter as tk
from tkinter import Canvas, StringVar, Toplevel, ttk
from tkinter.constants import CENTER, TOP
import tkinter.messagebox
from Model.setup_data import *
import threading
import View.myshallowFindings as shallow
#import View.Shallow_view as shallow
import View.mydeepFindings as deep
import View.mydeep_shallowFinding as deepShallow
import View.mydeep_deepFindings as deepDeep
class intro_menu:
    data_canvas = None
    data_status = None
    data = None
    root = None
    current_frame = None
    feature_type = None
    classification_type = None
    counter_screen = 0

    def __init__(self):
        self.data_canvas = None
        self.data_status = None
        self.data = {}
        self.root = None
        self.current_frame = None
        self.feature_type = None
        self.classification_type = None
        self.counter_screen = 0
        self.create_main_window()
    # COmmand Functions
    def initialSetUp(self):
        self.data['001'] = load_dataset()
        self.data['002'] = load_dataset_2()
        self.data_canvas.itemconfig(self.data_status, fill="green")


    def shallowSelect(self):
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.shallowSelection(self.root)
        self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
        self.counter_screen = 4

    def deepSelect(self):
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.DeepSelection(self.root)
        self.current_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.counter_screen = 4   

    def deepshallowSelect(self):
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.deepShallowSelection(self.root)
        self.current_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.counter_screen = 4  

    def deepdeepSelect(self):
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.deepDeepSelection(self.root)
        self.current_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.counter_screen = 4   

    def Findings_Frame_input(self, container):
        frame = ttk.Frame(container)
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(0, weight=4)
        ttk.Label(frame, text='My Findings:')
        tk.Button(frame, text='Shallow Learning', width = 40, command= (lambda: self.open_finding('Shallow Learning')))
        tk.Button(frame, text='Deep Learning', width = 40, command= (lambda: self.open_deepfinding('Deep Learning')))
        tk.Button(frame, text='Combination of Shallow and Deep Learning', width = 40, command= (lambda: self.open_deepshallowfinding('Deep-Shallow Learning')))
        tk.Button(frame, text='Combinaton of Deep Learning', width = 40, command= (lambda: self.open_deepdeepfinding('Deep-Deep Learning')))
        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)

        return frame

    
#Frame for Deep and Shallow Selection
    def open_deepdeepfinding(self, modeltype):
       
        if self.data_canvas.itemcget(self.data_status, 'fill') == 'green':     
            deepDeep.mydeep_deepFindings(self.data, modeltype, self.feature_type, self.classification_type)
        else:
            tkinter.messagebox.showwarning(title='Error', message='The dataset has not been loaded')


    #Frame for Deep and Shallow Selection
    def open_deepshallowfinding(self, modeltype):
       
        if self.data_canvas.itemcget(self.data_status, 'fill') == 'green':     
            deepShallow.mydeep_shallowFindings(self.data, modeltype, self.feature_type, self.classification_type)
        else:
            tkinter.messagebox.showwarning(title='Error', message='The dataset has not been loaded')


    # Frame for Deep Selection:
    def open_deepfinding(self, modeltype):
       
        if self.data_canvas.itemcget(self.data_status, 'fill') == 'green':     
            deep.mydeepFindings(self.data, modeltype, self.feature_type, self.classification_type)
        else:
            tkinter.messagebox.showwarning(title='Error', message='The dataset has not been loaded')
 
    # Frame for Shallow Selection:
    def open_finding(self, modeltype):
        if self.data_canvas.itemcget(self.data_status, 'fill') == 'green':     
            shallow.myshallowFindings(self.data, modeltype, self.feature_type, self.classification_type)
        else:
            tkinter.messagebox.showwarning(title='Error', message='The dataset has not been loaded')  


    #Frame to classification type:
        #for shallow
    def get_findings_model(self, num):
        self.classification_type = num
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.Findings_Frame_input(self.root)
        self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
        self.counter_screen = 3
        #for Deep

    def classification_frame(self, container):
        frame = ttk.Frame(container)
        ttk.Label(frame, text='Select Classification Type :', font = (10))
        tk.Button(frame, text='Binary Class', width = 20, command= (lambda: self.get_findings_model(0)))
        tk.Button(frame, text='Multi-Class', width = 20, command= (lambda: self.get_findings_model(1)))
        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)
        return frame
    # FIndings feature list frame

    def selectClassType(self, num):
        self.feature_type = num
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.classification_frame(self.root)
        self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
        self.counter_screen = 2
        

    def open_feature_list(self, container):
        frame = ttk.Frame(container)

        ttk.Label(frame, text='Select Feature Subset :', font = (10))
        tk.Button(frame, text='All Features (34)', width = 40, command= (lambda: self.selectClassType(0)))
        tk.Button(frame, text='Without IP Features (20)', width = 40, command= (lambda: self.selectClassType(1)))
        tk.Button(frame, text='Top 7 features with IP Features (CFS)', width = 40, command= (lambda: self.selectClassType(2)))
        tk.Button(frame, text='Top 6 features without IP Features (CFS)', width = 40, command= (lambda: self.selectClassType(3)))
        tk.Button(frame, text='Top 11 features with IP Features (IG)', width = 40, command= (lambda: self.selectClassType(4)))
        tk.Button(frame, text='Top 9 features without IP Features (IG)', width = 40, command= (lambda: self.selectClassType(5)))
        for widget in frame.winfo_children():
            widget.grid(padx=5, pady=5)
        return frame
    # Frame shown in introduction

    def selectfeaturetype(self):    
        
        self.current_frame.place_forget()
        self.current_frame.destroy()
        self.current_frame = self.open_feature_list(self.root)
        self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
        self.counter_screen = 1

    def intro_frame(self, container):
        frame = ttk.Frame(container)

        ttk.Label(frame, text='Intrusion Detection Using AI :',  font=( 12))
        tk.Button(frame, text='My Findings', width = 20, command= self.selectfeaturetype)
        #tk.Button(frame, text='Train Custom Models', width = 20)

        for widget in frame.winfo_children():
            widget.grid(padx=7, pady=7)
        return frame

    #Back to previous functions
    def Back_findings(self):
        if self.counter_screen == 1:
            self.current_frame.place_forget()
            self.current_frame.destroy()
            self.current_frame = self.intro_frame(self.root)
            self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
            self.counter_screen = 0  
        elif self.counter_screen == 2:
            self.current_frame.place_forget()
            self.current_frame.destroy()
            self.current_frame = self.open_feature_list(self.root)
            self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
            self.counter_screen = 1
        elif self.counter_screen == 3:
            self.current_frame.place_forget()
            self.current_frame.destroy()
            self.current_frame = self.classification_frame(self.root)
            self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
            self.counter_screen = 2
        elif self.counter_screen == 4:
            self.current_frame.place_forget()
            self.current_frame.destroy()
            self.current_frame = self.Findings_Frame_input(self.root)
            self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
            self.counter_screen = 3
        else:
            print('error')

    def create_main_window(self):
        global data_status, root, current_frame, data_canvas
        # root window
        self.root = tk.Tk()
        self.root.title('GUI')
        self.root.geometry('500x400')
        self.root.resizable(0, 0)
        # windows only (remove the minimize/maximize button)
        #root.attributes('-toolwindow', True)
        
        # layout on the root window
        
        tk.Button(self.root, text='<-- BACK', width = 10, command= self.Back_findings).place(relx = 0.1, rely = 0.05, anchor='n')
        ttk.Label(self.root, text = "DataSet Status --->").place(relx = 0.75, rely = 0.05, anchor='n')
        self.data_canvas =  Canvas(self.root, width=15, height=15)
        self.data_status = self.data_canvas.create_oval(15, 15, 5, 5)
        self.data_canvas.itemconfig(self.data_status, fill="red")
        self.data_canvas.place(relx = 0.95, rely = 0.05, anchor='n')
        self.current_frame = self.intro_frame(self.root)
        self.current_frame.place(relx=0.5, rely=0.6, anchor=CENTER)
        x = threading.Thread(target=self.initialSetUp, args=())
        x.start()

        self.root.mainloop()


