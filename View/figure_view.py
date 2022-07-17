import tkinter as tk
from tkinter import *
from tkinter import ttk

import matplotlib.backends.backend_tkagg as tkagg

def create_plot_window(figure):
    root = tk.Tk()
    root.title('Plot')
    root.geometry('1200x700')
    root.resizable(0, 0)
    canvasFrame = ttk.Frame(root, height= root.winfo_height()-200, width= root.winfo_width()-100)
    canvasFrame.columnconfigure(0,weight=1)
    canvasFrame.rowconfigure(0,weight=1)
    
    chart_type = tkagg.FigureCanvasTkAgg(figure, canvasFrame)
    #chart_type.draw()
    xbar = ttk.Scrollbar(canvasFrame, orient=HORIZONTAL)
    ybar = ttk.Scrollbar(canvasFrame, orient=VERTICAL)
    xbar.config(command=  chart_type.get_tk_widget().xview)
    ybar.config(command=  chart_type.get_tk_widget().yview)
    ybar.grid(row=0, column=1, sticky='ns')
    xbar.grid(row=1, column=0, sticky='ew')
    chart_type.get_tk_widget().config(xscrollcommand= xbar.set, yscrollcommand= ybar.set)
    chart_type.get_tk_widget().grid(row=0, column=0, sticky=tk.W)
    tkagg.NavigationToolbar2Tk(chart_type, root)
    canvasFrame.pack()
    #figure.savefig('test.png', bbox_inches='tight')
    root.mainloop()
    return True
