import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk


# Load the model and tokenizer
model = load_model('image_captioning_model.h5')  # Or 'saved_model/image_captioning_model' for TensorFlow SavedModel format
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max length of caption
maxlen = 34  # You should use the same value as during training

# Define a function to preprocess the image and predict the caption
def preprocess_image(image_path):
    npix = 224
    target_size = (npix, npix, 3)
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    return image.reshape((1,) + image.shape)

# Function to predict caption
def predict_caption(image):
    in_text = 'startseq'

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=maxlen)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "endseq":
            break
    return in_text

# Function to open file dialog and upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        # Display image in Tkinter window
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize image for display
        img = ImageTk.PhotoImage(image)

        panel.config(image=img)
        panel.image = img

        # Preprocess image
        image_array = preprocess_image(file_path)

        # Generate caption
        caption = predict_caption(image_array)
        caption = caption.replace("startseq", "").replace("endseq", "")

        # Display caption
        caption_label.config(text="Caption: " + caption)

# Create the main window
root = tk.Tk()
root.title("Image Captioning")

# Add a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Create an area to display the uploaded image
panel = tk.Label(root)
panel.pack()

# Label for displaying the predicted caption
caption_label = tk.Label(root, text="Caption will appear here", wraplength=400)
caption_label.pack()

# Run the Tkinter event loop
root.mainloop()
