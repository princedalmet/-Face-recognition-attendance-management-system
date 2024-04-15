from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = '/home/princeton/Downloads/captone/Final_project/Face-Recognition-based-Attendance-System_model.h5'
model = load_model(model_path)

def detect_person(image_path, model):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Assuming input shape is (224, 224, 3)
    img = img / 255.0  # Normalize pixel values

    # Reshape the image to match the input shape expected by the model
    img = np.expand_dims(img, axis=0)

    # Predict the label for the image
    prediction = model.predict(img)

    # Assuming your model outputs softmax probabilities, find the index with the highest probability
    predicted_label_index = np.argmax(prediction)

    # Map the index to the actual label
    labels = ['label1', 'label2', 'label3', 'label4', 'label5']  # Replace with your actual labels
    predicted_label = labels[predicted_label_index]

    return predicted_label

def save_to_excel(image_path, status):
    # Load the image filename without extension as ID
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # Create or load existing Excel file
    excel_file = 'attendance.xlsx'
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(columns=['ID', 'Status'])

    # Append new row with image ID and status
    df = df.append({'ID': image_id, 'Status': status}, ignore_index=True)

    # Save DataFrame to Excel file
    df.to_excel(excel_file, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_presence', methods=['POST'])
def check_presence():
    # Receive image file from frontend
    image_file = request.files['image']

    # Save image to a temporary location
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    # Detect person in the image
    status = detect_person(temp_image_path, model)

    # Save to Excel
    save_to_excel(temp_image_path, status)

    # Return status as JSON response
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(debug=True, port=5004)

