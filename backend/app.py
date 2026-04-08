import os
import uuid
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import predict_face_shape
from recommendations.hairstyles import get_recommendations

app = Flask(__name__, static_folder="../frontend")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route("/predict", methods=["POST"])
def predict():
    # Validate image
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only JPG and PNG files are allowed."}), 400

    # Validate gender
    gender = request.form.get("gender", "").lower().strip()
    if gender not in ("male", "female"):
        return jsonify({"error": "Please select a gender."}), 400

    # Get optional preference inputs (with safe defaults)
    hair_type   = request.form.get("hair_type",   "any").lower().strip()
    length_pref = request.form.get("length_pref", "medium").lower().strip()
    maintenance = request.form.get("maintenance", "low").lower().strip()

    # Save temp file
    ext       = file.filename.rsplit(".", 1)[1].lower()
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file.save(temp_path)

    try:
        prediction = predict_face_shape(temp_path)
        face_shape = prediction["face_shape"]

        recommendations = get_recommendations(
            face_shape  = face_shape,
            gender      = gender,
            hair_type   = hair_type,
            length_pref = length_pref,
            maintenance = maintenance
        )

        return jsonify({
            "face_shape":      face_shape,
            "confidence":      prediction["confidence"],
            "all_scores":      prediction["all_scores"],
            "gender":          gender,
            "preferences": {
                "hair_type":   hair_type,
                "length_pref": length_pref,
                "maintenance": maintenance
            },
            "recommendations": recommendations
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("Starting Hairstyle Recommender — http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)