from cProfile import label
from cgitb import text
import tkinter as tk
from PIL import Image, ImageTk
import os
root = tk.Tk()
ws = root.winfo_screenwidth()
wh = root.winfo_screenheight()
w = 700
h = 500
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
root.geometry("%dx%d+%d+%d" % (w, h, x, y))
root.title("Findfinity")
root.iconbitmap(r"Images\ico.ico")
root.tk_setPalette(background="#6b6b6b", foreground="#fdfcff", activeBackground="#6b6b6b", activeForeground="#6b6b6b")
frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=10, pady=10)
image_path = "Images/logo.png"
original_image = Image.open(image_path)
desired_width = 250
aspect_ratio = original_image.width / original_image.height
desired_height = int(desired_width / aspect_ratio)
resized_image = original_image.resize((desired_width, desired_height), Image.ANTIALIAS)
tk_image = ImageTk.PhotoImage(resized_image)
image_label = tk.Label(frame, image=tk_image)
image_label.pack(anchor="n")
def run():
    root.destroy()
    os.system('python main.py')
Button=tk.Button(frame,text="Start",font=("Arial",30),relief='flat', highlightthickness=0, bd=0,bg="#0c6098",fg="white",command=run)
Button.place(x=270,y=300)

names=tk.Label(text="Ahmad Abu Alzahab ,Amru Buzu , Mhd Assad, Simon Azar ",font=("Arial",16))
names.place(x=60,y=400)



root.mainloop()
