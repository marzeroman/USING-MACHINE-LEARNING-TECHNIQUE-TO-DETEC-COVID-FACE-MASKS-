from tkinter import *
from tkinter import filedialog
import subprocess

def detect_live_video():
    subprocess.call(["python", "DetectMaskFunctionByVideo.py"])

def detect_picture():
    file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    if file_path:
        subprocess.call(["python", "PictureMaskDetection.py", file_path])

def create_interface():
    root = Tk()
    root.title("Face Mask Detection")
    root.geometry("300x200")

    label = Label(root, text="Choose detection method:")
    label.pack(pady=10)

    live_video_button = Button(root, text="Live Video", command=detect_live_video)
    live_video_button.pack()

    picture_button = Button(root, text="Static Picture", command=detect_picture)
    picture_button.pack()

    root.mainloop()

# Create the user interface
create_interface()
