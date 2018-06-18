import tkinter as tk 
import matplotlib.pyplot as plt
import numpy as np
from Rn import NN

from tkinter import ttk, filedialog, messagebox, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk   

spaceBetweenElements = 7
spaceTop = 10

class PagesController(tk.Tk):
    def __init__(self, *args, **kargs):
        tk.Tk.__init__(self, *args, **kargs)
        self.geometry("1000x650")
        self.title("Neural Network")
        self.minsize(width = 1000, height = 650)
        container = tk.Frame(self)
        container.pack(side = "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        
        self.frames = {}
        for F in (StartPage, NNPage, PaintRn):
            
            frame = F (container, self)
            self.frames[F] = frame
        
            frame.grid(row = 0 , column = 0, sticky = "nsew")
        self.show_frame(StartPage)
        
    def show_frame(self, count, data = None):
         frame = self.frames[count]
         
         if count is NNPage or count is PaintRn:
             frame.SetData(data)
         frame.tkraise()
         

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text = "Please, add a .csv file to training the neural network!")
        label.pack(pady = (spaceTop ,0))    
        fileButton = ttk.Button(self, text="Choose file", command=self.FileChooser)
        fileButton.pack(pady = (spaceBetweenElements, 0))
        self.nnButton = ttk.Button(self, text="Start the process", command= lambda: controller.show_frame(NNPage, self.nn), state = tk.DISABLED)
        self.nnButton.pack(pady =(spaceBetweenElements, 0))
        
    def FileChooser(self):
        filename = filedialog.askopenfilename(title = "Choose a file", filetypes = [('csv files', '.csv')])
        self.nn = NN()
        self.nn.LoadData(filename)
        self.nnButton['state'] = 'normal'

class NNPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid(row = 0 , column = 0, sticky = "nsew")
        self.nn = None
        self.MainWindow()
        
    def SetData(self, data):
        self.nn = data
        
    def MainWindow(self):
       
        for widget in self.winfo_children():
            widget.destroy()
        
        self.accButton = ttk.Button(self, text = "Show accuracy", command = self.ShowPredictionPercent, state = tk.DISABLED if self.nn is None else 'normal')
        self.accButton.pack(pady =(spaceTop, 0))
        
        self.startRn = ttk.Button(self, text = "Start", command = self.StartNN)
        self.startRn.pack(pady =(spaceBetweenElements, 0))
        
        self.afiseazaRn = ttk.Button(self, text = "Show the neural netowrk arhitecture", command = lambda: self.controller.show_frame(PaintRn, self.nn ))
        self.afiseazaRn.pack(pady =(spaceBetweenElements, 0))
        
        
    def StartNN(self):
        if self.nn is not None:
            answer = messagebox.askokcancel("Confirmation!", "After you start to train the neural network, the aplication will freeze for a while. After this process the \"Show accuracy\" button will be active. Do you want to continue?")
            if answer:
                self.nn.Initialization()
                self.accButton['state'] = "normal"
        
    def ShowPredictionPercent(self):
       self.accButton.destroy()
       self.startRn.destroy()
       self.afiseazaRn.destroy()
       button = ttk.Button(self, text = "Back", command = self.MainWindow)
       button.pack(pady =(spaceTop, spaceBetweenElements))
       labels = ('Correct result for training', 'Wrong result for training' , 'Correct result for testing', 'Wrong result for testing')
       percent = self.nn.percentsArray[0]
       
       f, a =  plt.subplots()
       tects = a.bar(np.arange(4),[percent[0] , 100 - percent[0], percent[1] , 100 - percent[1]], width = 0.2)
       plt.xticks(np.arange(4), labels)

       canvas = FigureCanvasTkAgg(f, self)
       canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True )
       toolbar = NavigationToolbar2Tk( canvas, self)
       toolbar.update()
       canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
    
       canvas.draw()
    

class PaintRn(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        button = ttk.Button(self, text = "Back", command = self.BackToPreviouse)
        button.pack(pady =(spaceTop, 0))
        
    def BackToPreviouse(self):
        self.controller.show_frame(NNPage, self.data)
        self.canvas.pack_forget()
        
    def SetData(self, data):
        self.data = data
        dataLenght = [len(data.df.columns) - 1, len(data.setClassList)]
        self.canvas = Canvas(self.controller, width = self.controller.winfo_width(), height = self.controller.winfo_height())
        self.PaintNN(dataLenght)
         
    def PaintNN(self, dataLenght):
        
        inputLenght, classLenght = dataLenght
        maxim = np.maximum(inputLenght, classLenght)
        w,h = self.controller.winfo_width(), self.controller.winfo_height()
        h = h - 100
        spaceY = 20
        spaceXLayer1 = 100
        spaceTop = 15
        diameter = ( h - spaceTop - spaceY) / maxim - spaceY 
       
        positionsLayer1 = list()
        positionsLayer2 = list()
      
        for input in np.arange(inputLenght):
            yPos = (input) * ( diameter + spaceY) + spaceTop
            positionsLayer1.append(yPos)
            self.CreateCircle(self.canvas, spaceXLayer1, yPos, diameter)
    
        spaceXLayer2 = w / 2 - diameter / 2
        spaceY = h / classLenght - diameter
        spaceTop = spaceY / 2
        for input in np.arange(classLenght):
            yPos = (input) * ( diameter + spaceY) + spaceTop
            positionsLayer2.append(yPos)
            self.CreateCircle(self.canvas, spaceXLayer2, yPos, diameter)
           
        self.CreateCircle(self.canvas, w -200 , h / 2 - diameter / 2 , diameter)  
       
        for layer1 in positionsLayer1:
            for layer2 in positionsLayer2:
                self.canvas.create_line(spaceXLayer1 + diameter , layer1 + diameter / 2 , spaceXLayer2, layer2 + diameter/2)
       
        for layer2 in positionsLayer2: 
            self.canvas.create_line(spaceXLayer2 + diameter, layer2 + diameter / 2, w-200, h / 2 )

        self.canvas.update()
        self.canvas.pack()
        
    def CreateCircle(self,canvas, x,y, diameter):
       canvas.create_oval(x,y, x + diameter, y + diameter)

app = PagesController()
app.mainloop()



