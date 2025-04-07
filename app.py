from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import jsonify
import requests
import io


app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

models= {}

# Load model from GCS URL
MODEL_URL = "https://storage.googleapis.com/srihari-2704/xmax_xgb_model.pkl"

response = requests.get(MODEL_URL)
models["xmax"] = joblib.load(io.BytesIO(response.content))


MODEL_URL = "https://storage.googleapis.com/srihari-2704/xmin_xgb_model.pkl"

response = requests.get(MODEL_URL)
models["xmin"] = joblib.load(io.BytesIO(response.content))

MODEL_URL = "https://storage.googleapis.com/srihari-2704/ymax_xgb_model.pkl"

response = requests.get(MODEL_URL)
models["ymax"] = joblib.load(io.BytesIO(response.content))

MODEL_URL = "https://storage.googleapis.com/srihari-2704/ymin_xgb_model.pkl"

response = requests.get(MODEL_URL)
models["ymin"] = joblib.load(io.BytesIO(response.content))


# Feature extraction using SIFT
def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((1, 128))  # fallback
    descriptors = descriptors.flatten()
    if descriptors.shape[0] < 12800:
        descriptors = np.pad(descriptors, (0, 12800 - descriptors.shape[0]))
    else:
        descriptors = descriptors[:12800]
    return descriptors.reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', uploaded_image=None, message="No file uploaded.")

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and process image
        image = cv2.imread(filepath)
        if image is None:
            return render_template('index.html', uploaded_image=None, message="Invalid image file.")

        features = extract_sift_features(image)

        # Predict bounding box
        bbox = [int(models[label].predict(features)[0]) for label in ['xmin', 'ymin', 'xmax', 'ymax']]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        output_filename = f"output_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, image)

        # Pass the static-relative path for frontend
        output_url = request.host_url.rstrip('/') + '/static/' + output_filename
        return jsonify({'image_path': output_url})

    return render_template('index.html', uploaded_image=None)

if __name__ == '__main__':
    app.run(debug=True)
