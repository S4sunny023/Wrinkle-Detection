import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle

class WrinkleDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        style = ThemedStyle(window)
        style.set_theme("arc") 

        self.face_cascade = cv2.CascadeClassifier("F:\\Wrinkles_detection\\haarcascade_frontalface_default.xml")

        self.canvas = tk.Canvas(window, width=800, height=600)
        self.canvas.pack()

        self.btn_load = ttk.Button(window, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_quit = ttk.Button(window, text="Quit", command=self.window.destroy)
        self.btn_quit.pack(side=tk.RIGHT, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  
            if img is not None:
                self.detect_wrinkles(img)
            else:
                print("Error loading image")

    def detect_wrinkles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            cropped_img = frame[y:y + h, x:x + w]
            edges = cv2.Canny(cropped_img, 50, 150)  
            number_of_edges = np.count_nonzero(edges)

            if number_of_edges > 2000:  
                text = "Wrinkle Found"
                color = (0, 255, 0) 
            else:
                text = "No Wrinkle Found"
                color = (0, 0, 255)  

            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) 

        self.display_image(frame)

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((800, 600)) 
        photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  

if __name__ == "__main__":
    root = tk.Tk()
    app = WrinkleDetectionApp(root, "Wrinkle Detection Application")
    root.mainloop()
