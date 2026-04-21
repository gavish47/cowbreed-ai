from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import json
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("models/testmodel.h5")

with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

breed_info = {
    "Sahiwal": {
        "origin": "Punjab (India/Pakistan)",
        "description": "High milk yield and heat tolerant breed"
    },
    "Lakhimi": {
        "origin": "Assam (India)",
        "description": "Indigenous breed adapted to humid climate"
    },
    "Siri": {
        "origin": "Himalayan region",
        "description": "Strong breed used for draft and milk"
    },
    "Umblachery": {
        "origin": "Tamil Nadu (India)",
        "description": "Used for draught work"
    }
}

def predict_breed(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))

    # 🔥 IMPORTANT FIX (MobileNet preprocessing)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    top_index = np.argmax(preds)
    confidence = float(preds[top_index] * 100)

    breed = index_to_class[top_index]

    # Top 3 predictions
    top_indices = preds.argsort()[-3:][::-1]

    others = []
    for i in top_indices[1:]:
        others.append({
            "breed": index_to_class[i],
            "conf": round(float(preds[i] * 100), 2)
        })

    info = breed_info.get(breed, {"origin": "Unknown", "description": "N/A"})

    return {
        "breed": breed,
        "confidence": round(confidence, 2),
        "origin": info["origin"],
        "description": info["description"],
        "others": others
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = predict_breed(filepath)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
