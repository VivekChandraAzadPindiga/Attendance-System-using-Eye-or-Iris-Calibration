import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Correct paths to the model file and classes file
model_path = r"C:\Users\Public\Iris Project\iris_attendance_cnn_segmentation_model.h5"
classes_path = r"C:\Users\Public\Iris Project\classes.npy"

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(classes_path)

# Initialize the list of candidates in the hall
if 'candidates' not in st.session_state:
    st.session_state.candidates = []

def predict(image):
    image = cv2.resize(image, (128, 128))
    image = np.reshape(image, (1, 128, 128, 1))
    image = image / 255.0
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]

st.title("Iris Attendance System")

# Reset button to clear the list
if st.button('Reset List'):
    st.session_state.candidates = []

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(image, channels="GRAY")
    result = predict(image)
    st.write(f"Predicted Subject ID: {result}")

    # Update the list of candidates
    if result in st.session_state.candidates:
        st.session_state.candidates.remove(result)
        st.write(f"{result} removed from the hall.")
    else:
        st.session_state.candidates.append(result)
        st.write(f"{result} added to the hall.")

# Display the list of candidates in a table
print(1)
st.write("Candidates in the hall:")
candidates_df = pd.DataFrame(st.session_state.candidates, columns=["Subject ID"])
st.table(candidates_df)
