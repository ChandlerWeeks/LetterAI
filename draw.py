import tkinter as tk
from PIL import Image, ImageOps
import os
from train import ConvNet
import torch
import numpy as np
import io

class DrawingBoard:
  def __init__(self, master):
    self.model = ConvNet()
    self.model.load_model()

    self.i = 0
    # Create a frame for the canvas and buttons
    self.frame = tk.Frame(master)
    self.frame.pack()

    # Create the canvas
    self.canvas = tk.Canvas(self.frame, width=200, height=200, bg='white')
    self.canvas.grid(row=0, column=0, pady=10, padx=10)

    # Bind the draw function to mouse motion
    self.canvas.bind("<B1-Motion>", self.draw)

    # Create two buttons
    self.button1 = tk.Button(self.frame, text="Guess Letter", command=self.button1_func)
    self.button1.grid(row=0, column=1, padx=10, pady=10, ipady=5)
    
    self.button2 = tk.Button(self.frame, text="Clear Canvas", command=self.button2_func)
    self.button2.grid(row=0, column=2, padx=10, pady=10, ipady=5)

    self.guess_label = tk.Label(self.frame, text="Guess: ")
    self.guess_label.grid(row=1, column=1)  # Adjust the row and column as needed

  def draw(self, event):
    r=5
    self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='black', outline='black')

  def save_image(self):
    ps = self.canvas.postscript(colormode='color')
    im = Image.open(io.BytesIO(ps.encode('utf-8')))
    im = im.resize((28, 28))

    im = ImageOps.invert(im)

    im.save(f'drawing.jpg')
    self.i+=1

  def load_image(self):
    image = Image.open('drawing.jpg')
    image = image.convert('L')
    image = image.rotate(-90)
    image_array = np.array(image)
    image_data = image_array.reshape(1, 1, 28, 28)
    image_tensor = torch.from_numpy(image_data).float()
    return image_tensor
  
  def button1_func(self):
    self.save_image()
    self.make_guess()

  def button2_func(self):
    self.canvas.delete("all")

  def on_close(self):
    if os.path.exists("drawing.eps"):
      os.remove("drawing.eps")

      root.destroy()

  def make_guess(self):
    img = self.load_image()

    # Make the prediction
    output = self.model.forward(img)
    self.guess_label.config(text=f"Guess: {self.model.mapping[output.argmax().item()]}")

root = tk.Tk()
db = DrawingBoard(root)
root.mainloop()