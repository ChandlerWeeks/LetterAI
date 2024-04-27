import tkinter as tk
from PIL import Image, ImageDraw
import os
from train import ConvNet
import torch
import numpy as np

class DrawingBoard:
  def __init__(self, master):
    self.model = ConvNet()
    self.model.load_model()

    master.protocol("WM_DELETE_WINDOW", self.on_close)
    self.i = 0
    # Create a frame for the canvas and buttons
    self.frame = tk.Frame(master)
    self.frame.pack()

    # Create the canvas
    self.canvas = tk.Canvas(self.frame, width=250, height=250, bg='white')
    self.canvas.grid(row=0, column=0, pady=10, padx=10)

    # Bind the draw function to mouse motion
    self.canvas.bind("<B1-Motion>", self.draw)

    # Create two buttons
    self.button1 = tk.Button(self.frame, text="Guess Letter", command=self.button1_func)
    self.button1.grid(row=0, column=1, padx=10, pady=10, ipady=5)
    
    self.button2 = tk.Button(self.frame, text="Clear Canvas", command=self.button2_func)
    self.button2.grid(row=0, column=2, padx=10, pady=10, ipady=5)

  def draw(self, event):
    r=2
    self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='black')

  def button1_func(self):
    self.canvas.postscript(file="drawing.eps", colormode='color')
    img = Image.open("drawing.eps")

    # resize image to be processed by neural network
    img = img.resize((28, 28))

    img.save(f"drawing_{self.i}.png", "png")
    self.make_guess()
    self.i += 1

    #TODO: add code to predict the letter

  def button2_func(self):
    self.canvas.delete("all")

  def on_close(self):
    if os.path.exists("drawing.eps"):
      os.remove("drawing.eps")

      root.destroy()

  def make_guess(self):
    # Load the image
    img = Image.open(f"drawing_{self.i}.png")
    img = img.convert("L")

    # Convert the image to a tensor
    img = torch.tensor(np.array(img), dtype=torch.float32)
    img = img.view(1, 1, 28, 28)

    # Make the prediction
    output = self.model.forward(img)
    print(f"Prediction: {self.model.mapping[torch.argmax(output).item()]}")

root = tk.Tk()
db = DrawingBoard(root)
root.mainloop()