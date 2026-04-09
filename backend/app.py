import os
import uuid
import sys
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import predict_face_shape
from recommendations.hairstyles import get_recommendations
from database import db, User, Favorite

app = Flask(__name__, static_folder="../frontend")
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hair_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'super-secret-key-for-jwt-12345'
db.init_app(app)

with app.app_context():
    db.create_all()

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

# ─── Auth & DB Routes ───────────────────────────────────────────────

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"message": "Token is missing!"}), 401
        try:
            token = token.split(" ")[1] # Bearer Token
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data["user_id"]).first()
        except Exception as e:
            return jsonify({"message": "Token is invalid!"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"message": "Missing credentials!"}), 400
        
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"message": "User already exists!"}), 400
        
    hashed_password = generate_password_hash(data["password"], method="pbkdf2:sha256")
    new_user = User(username=data["username"], password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    # Return a token immediately so they are logged in
    token = jwt.encode({"user_id": new_user.id, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)}, app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"message": "Account created!", "token": token, "username": new_user.username}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"message": "Missing credentials!"}), 400
        
    user = User.query.filter_by(username=data["username"]).first()
    if not user or not check_password_hash(user.password_hash, data["password"]):
        return jsonify({"message": "Invalid username or password!"}), 401
        
    token = jwt.encode({"user_id": user.id, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)}, app.config["SECRET_KEY"], algorithm="HS256")
    return jsonify({"message": "Login successful!", "token": token, "username": user.username}), 200

@app.route("/api/favorites", methods=["GET"])
@token_required
def get_favorites(current_user):
    favs = Favorite.query.filter_by(user_id=current_user.id).all()
    output = []
    for fav in favs:
        output.append({"id": fav.id, "hairstyle_name": fav.hairstyle_name})
    return jsonify({"favorites": output}), 200

@app.route("/api/favorites/add", methods=["POST"])
@token_required
def add_favorite(current_user):
    data = request.get_json()
    hairstyle_name = data.get("hairstyle_name")
    
    # Check if already exists
    if Favorite.query.filter_by(user_id=current_user.id, hairstyle_name=hairstyle_name).first():
        return jsonify({"message": "Already favorited"}), 200
        
    new_fav = Favorite(user_id=current_user.id, hairstyle_name=hairstyle_name)
    db.session.add(new_fav)
    db.session.commit()
    return jsonify({"message": "Added to favorites!"}), 201

@app.route("/api/favorites/remove", methods=["POST"])
@token_required
def remove_favorite(current_user):
    data = request.get_json()
    hairstyle_name = data.get("hairstyle_name")
    
    fav = Favorite.query.filter_by(user_id=current_user.id, hairstyle_name=hairstyle_name).first()
    if fav:
        db.session.delete(fav)
        db.session.commit()
    return jsonify({"message": "Removed from favorites!"}), 200


@app.route("/analyze-hair", methods=["POST"])
def analyze_hair_route():
    """Lightweight endpoint: detect hair length + type only (no CNN, no recommendations)."""
    from model.mediapipe_analysis import analyze_hair as _analyze_hair
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file."}), 400

    ext       = file.filename.rsplit(".", 1)[1].lower()
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file.save(temp_path)
    try:
        result = _analyze_hair(temp_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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
        
        auto_length = prediction.get("auto_hair_length", "medium")
        auto_type = prediction.get("auto_hair_type", "any")
        
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
            "cnn_scores":      prediction.get("cnn_scores", {}),
            "geo_scores":      prediction.get("geo_scores", {}),
            "detected_hair": {
                "length": auto_length,
                "type": auto_type
            },
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