from flask import Flask, render_template, request, Response, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load MobileNetV2
print("Loading MobileNetV2...")
model = MobileNetV2(weights="imagenet")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Start camera
camera = cv2.VideoCapture(0)
print("Camera opened?", camera.isOpened())


# --------------------------------------------------
# LIVE CAMERA STREAM
# --------------------------------------------------
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# CAPTURE & PREDICT (ONLY LABEL)
# --------------------------------------------------
@app.route("/capture")
def capture():
    try:
        ret, frame = camera.read()
        if not ret:
            return jsonify({"error": "Camera capture failed"}), 500

        img_path = os.path.join(UPLOAD_FOLDER, "captured.jpg")
        cv2.imwrite(img_path, frame)

        # MobileNet prediction
        resized = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        result = decode_predictions(preds, top=1)[0][0]

        label = str(result[1])   # ONLY LABEL

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# UPLOAD & PREDICT (ONLY LABEL)
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    result = decode_predictions(preds, top=1)[0][0]

    label = str(result[1])

    return render_template(
        "index.html",
        label=label,
        img_path=filepath
    )


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
