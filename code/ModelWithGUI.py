# Imports
import streamlit as st
import tensorflow as tf
import os
import numpy as np
from PIL import Image


class_names = ["glioma", "meningioma", "no tumor", "pituitary"]

## Page Title
st.set_page_config(page_title="Brain Tumor Detection")
st.title("Brain Tumor Detection")
st.markdown("---")

## Sidebar
st.sidebar.header("Models")
display = ("Select a Model", "ResNet50 Model")
options = list(range(len(display)))
value = st.sidebar.selectbox("Model", options, format_func=lambda x: display[x])
print(value)

if value == 1:
    tflite_interpreter = tf.lite.Interpreter(model_path="model.tflite")
    tflite_interpreter.allocate_tensors()


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_predictions(input_image):
    output_details = tflite_interpreter.get_output_details()
    set_input_tensor(tflite_interpreter, input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis=0)
    pred_class = class_names[tflite_model_prediction]
    return pred_class


## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    st.image(img)
    print(value)
    if value == 2 or value == 5:
        img = tf.image.convert_image_dtype(img, tf.uint8)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

if st.button("Get Predictions"):
    suggestion = get_predictions(input_image=img_array)
    st.success(suggestion)
