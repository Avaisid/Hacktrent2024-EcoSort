from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime, timezone
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError
import os

app = Flask(__name__)

# AWS S3 Credentials
# ACCESS_KEY = ''
# SECRET_KEY = ''
BUCKET_NAME = 'wastecollector'

# MongoDB Connection Configuration
CONNECTION_STRING = "mongodb+srv://dikshithreddymacherla:6bhGUbPrCHFkMtms@hacktrent.xjzzo.mongodb.net/?retryWrites=true&w=majority&appName=HackTrent"
client = MongoClient(CONNECTION_STRING)
db = client['EcoSort']  # Database name
collection = db['wasteImages']  # Collection name

# Load the trained waste classification model
model_path = 'C:/Users/diksh/Desktop/HackTrent/EcoSort AI/AI Classification/waste_classifier_model.h5'
model = load_model(model_path)

# AWS S3 Upload Function
def upload_to_aws(file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    try:
        s3.upload_fileobj(file, bucket, s3_file)
        return True
    except FileNotFoundError:
        return False
    except NoCredentialsError:
        return False

# Function to preprocess the image for classification
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to the input size for MobileNetV2
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

@app.route('/upload_to_s3', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Upload to AWS S3
    success = upload_to_aws(file, BUCKET_NAME, file.filename)
    if success:
        # Insert Metadata into MongoDB Collection
        new_entry = {
            "image_id": file.filename,
            "label": "pending",  # Status or label, updated after classification
            "timestamp": datetime.now(timezone.utc).isoformat()  # Timestamp when the image was uploaded
        }

        try:
            # Insert metadata entry into MongoDB
            result = collection.insert_one(new_entry)
            return jsonify({"message": "Upload Successful", "document_id": str(result.inserted_id)}), 200
        except Exception as e:
            return jsonify({"error": f"Upload successful but failed to save metadata: {str(e)}"}), 500
    else:
        return jsonify({"error": "Upload failed"}), 500

@app.route('/classify_waste', methods=['POST'])
def classify_waste():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Open the image and preprocess it
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)

        # Predict the class of the waste
        prediction = model.predict(processed_image)
        class_names = ['Biological', 'Metal', 'Paper', 'Plastic']
        category = class_names[np.argmax(prediction)]

        # Update MongoDB entry after classification
        collection.update_one(
            {"image_id": file.filename},
            {"$set": {"label": category, "classification_timestamp": datetime.now(timezone.utc).isoformat()}}
        )

        return jsonify({'category': category})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
