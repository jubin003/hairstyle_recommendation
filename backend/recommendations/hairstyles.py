"""
Content-Based Filtering using Cosine Similarity for hairstyle recommendations.

Each hairstyle is encoded as a feature vector:
[oval, round, square, heart, oblong, male, female, length(0-2), maintenance(0-2), hair_type(0-3)]

Scores range 0.0 - 1.0 (how suitable the hairstyle is for that feature)
Length:      0=short, 1=medium, 2=long
Maintenance: 0=low,   1=medium, 2=high
Hair type:   0=any,   1=straight, 2=wavy, 3=curly
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── Feature index constants ───────────────────────────────────────
FACE_OVAL    = 0
FACE_ROUND   = 1
FACE_SQUARE  = 2
FACE_HEART   = 3
FACE_OBLONG  = 4
GENDER_MALE  = 5
GENDER_FEM   = 6
LENGTH       = 7   # 0=short 1=medium 2=long
MAINTENANCE  = 8   # 0=low   1=medium 2=high
HAIR_TYPE    = 9   # 0=any   1=straight 2=wavy 3=curly

# ─── Hairstyle database ────────────────────────────────────────────
# Format: [oval, round, square, heart, oblong, male, female, length, maintenance, hair_type]
HAIRSTYLES = {
    # ── FEMALE STYLES ──────────────────────────────────────────────
    "Curtain Bangs": {
        "vector":      [0.9, 0.5, 0.7, 0.8, 0.95, 0.3, 0.95, 1, 1, 2],
        "description": "Soft parted bangs that frame the face and suit most face shapes.",
        "tip":         "Part in the middle and blow-dry outward for best results.",
        "gender":      "female"
    },
    "Wavy Lob": {
        "vector":      [0.9, 0.6, 0.5, 0.95, 0.5, 0.1, 0.95, 1, 1, 2],
        "description": "A long bob with soft waves — adds volume to the lower face.",
        "tip":         "Use a large barrel curling iron for soft natural waves.",
        "gender":      "female"
    },
    "Pixie Cut": {
        "vector":      [0.95, 0.4, 0.6, 0.5, 0.5, 0.1, 0.9, 0, 0, 1],
        "description": "Short and chic — highlights balanced facial features.",
        "tip":         "Works best with fine to medium hair texture.",
        "gender":      "female"
    },
    "Blunt Bob": {
        "vector":      [0.8, 0.4, 0.4, 0.7, 0.9, 0.1, 0.9, 1, 1, 1],
        "description": "Clean horizontal lines add width and shorten face length.",
        "tip":         "Keep ends blunt and straight for maximum effect.",
        "gender":      "female"
    },
    "Long Layers": {
        "vector":      [0.9, 0.9, 0.7, 0.6, 0.5, 0.1, 0.95, 2, 1, 0],
        "description": "Layers starting below the chin elongate and slim the face.",
        "tip":         "Ask for face-framing layers for extra effect.",
        "gender":      "female"
    },
    "Side-Swept Bangs": {
        "vector":      [0.85, 0.7, 0.8, 0.9, 0.6, 0.1, 0.95, 1, 1, 0],
        "description": "Diagonal bangs soften strong features and wide foreheads.",
        "tip":         "Keep bangs long and angled — never blunt.",
        "gender":      "female"
    },
    "High Top Knot": {
        "vector":      [0.9, 0.9, 0.5, 0.5, 0.6, 0.1, 0.9, 2, 0, 0],
        "description": "Adds height making the face appear longer and slimmer.",
        "tip":         "Keep the sides sleek and flat for best effect.",
        "gender":      "female"
    },
    "Beachy Waves": {
        "vector":      [0.95, 0.6, 0.8, 0.7, 0.6, 0.1, 0.9, 2, 1, 2],
        "description": "Effortless loose waves that suit balanced facial features.",
        "tip":         "Salt spray gives a natural, textured look.",
        "gender":      "female"
    },
    "Soft Curls": {
        "vector":      [0.85, 0.5, 0.9, 0.7, 0.6, 0.1, 0.9, 1, 2, 3],
        "description": "Soft curls reduce angular features and add femininity.",
        "tip":         "Use a diffuser to enhance natural curl pattern.",
        "gender":      "female"
    },
    "Sleek Straight": {
        "vector":      [0.95, 0.4, 0.5, 0.6, 0.4, 0.1, 0.85, 2, 1, 1],
        "description": "Clean straight hair showcases balanced proportions.",
        "tip":         "A centre part works especially well for oval faces.",
        "gender":      "female"
    },
    "Voluminous Waves": {
        "vector":      [0.8, 0.5, 0.6, 0.6, 0.9, 0.1, 0.9, 2, 2, 2],
        "description": "Wide waves add volume on the sides, reducing face length.",
        "tip":         "Use a large barrel and brush waves out for maximum volume.",
        "gender":      "female"
    },
    "Low Ponytail": {
        "vector":      [0.85, 0.6, 0.6, 0.9, 0.6, 0.1, 0.9, 2, 0, 0],
        "description": "A low loose ponytail keeps the top flat and frames the chin.",
        "tip":         "Let a few face-framing strands fall loose.",
        "gender":      "female"
    },

    # ── MALE STYLES ────────────────────────────────────────────────
    "Textured Quiff": {
        "vector":      [0.9, 0.7, 0.6, 0.9, 0.7, 0.95, 0.1, 0, 1, 0],
        "description": "Adds volume at the front while keeping sides short.",
        "tip":         "Use matte pomade for a natural finish.",
        "gender":      "male"
    },
    "Side Part Undercut": {
        "vector":      [0.9, 0.7, 0.7, 0.9, 0.6, 0.95, 0.1, 0, 1, 1],
        "description": "Clean sides with a defined part — sharp and versatile.",
        "tip":         "Keep the part deep and well-defined.",
        "gender":      "male"
    },
    "High Fade with Height": {
        "vector":      [0.85, 0.95, 0.7, 0.6, 0.5, 0.95, 0.1, 0, 1, 0],
        "description": "Tight sides with height on top elongates round faces.",
        "tip":         "A high skin fade creates the best contrast.",
        "gender":      "male"
    },
    "Pompadour": {
        "vector":      [0.9, 0.9, 0.6, 0.7, 0.5, 0.95, 0.1, 0, 2, 1],
        "description": "Vertical volume elongates the face significantly.",
        "tip":         "Keep sides very short to contrast the top volume.",
        "gender":      "male"
    },
    "Crew Cut": {
        "vector":      [0.95, 0.6, 0.7, 0.7, 0.7, 0.95, 0.1, 0, 0, 0],
        "description": "Clean timeless style that suits balanced features.",
        "tip":         "Works for all hair types — very low maintenance.",
        "gender":      "male"
    },
    "Buzz Cut": {
        "vector":      [0.85, 0.5, 0.6, 0.5, 0.7, 0.95, 0.1, 0, 0, 0],
        "description": "Ultra short — clean and low maintenance.",
        "tip":         "Works best with a well-defined jawline.",
        "gender":      "male"
    },
    "Textured Crop": {
        "vector":      [0.85, 0.6, 0.7, 0.7, 0.9, 0.95, 0.1, 0, 1, 0],
        "description": "Horizontal fringe breaks up face length visually.",
        "tip":         "Ask for a French crop with a textured finish.",
        "gender":      "male"
    },
    "Slicked Back Undercut": {
        "vector":      [0.95, 0.5, 0.6, 0.6, 0.5, 0.95, 0.1, 1, 2, 1],
        "description": "Highlights facial symmetry with a sharp polished look.",
        "tip":         "Use strong hold pomade and comb straight back.",
        "gender":      "male"
    },
    "Messy Fringe": {
        "vector":      [0.85, 0.6, 0.7, 0.9, 0.7, 0.9, 0.1, 1, 1, 0],
        "description": "Forward-swept fringe softens the forehead width.",
        "tip":         "Avoid a perfectly straight fringe — keep it casual.",
        "gender":      "male"
    },
    "Long Hair with Layers": {
        "vector":      [0.85, 0.6, 0.9, 0.7, 0.5, 0.9, 0.1, 2, 1, 0],
        "description": "Length past the jaw softens angular features.",
        "tip":         "Go for at least collar-length to see the effect.",
        "gender":      "male"
    },
    "Curtain Hair (Male)": {
        "vector":      [0.85, 0.6, 0.8, 0.8, 0.9, 0.9, 0.1, 1, 1, 2],
        "description": "Parted middle fringe adds width and softens the face.",
        "tip":         "Works great with medium length wavy hair.",
        "gender":      "male"
    },
    "Modern Quiff": {
        "vector":      [0.95, 0.7, 0.6, 0.7, 0.6, 0.95, 0.1, 0, 1, 0],
        "description": "Versatile stylish quiff that suits oval faces perfectly.",
        "tip":         "Works with straight or wavy hair.",
        "gender":      "male"
    },
}

# ─── Encoding maps ─────────────────────────────────────────────────
FACE_SHAPE_IDX = {"oval": 0, "round": 1, "square": 2, "heart": 3, "oblong": 4}
GENDER_IDX     = {"male": GENDER_MALE, "female": GENDER_FEM}
LENGTH_MAP     = {"short": 0, "medium": 1, "long": 2}
MAINTENANCE_MAP= {"low": 0, "medium": 1, "high": 2}
HAIR_TYPE_MAP  = {"any": 0, "straight": 1, "wavy": 2, "curly": 3}


def _build_user_vector(face_shape, gender, hair_type="any",
                       length_pref="medium", maintenance="low"):
    """
    Builds a 10-dimensional user preference vector.
    Face shape confidence is set to 1.0 for the detected shape, 0.0 for others.
    """
    vec = [0.0] * 10

    # Face shape — 1.0 for detected shape
    if face_shape in FACE_SHAPE_IDX:
        vec[FACE_SHAPE_IDX[face_shape]] = 1.0

    # Gender — 1.0 for selected gender
    vec[GENDER_IDX[gender]] = 1.0

    # Preferences — normalised to 0-1
    vec[LENGTH]      = LENGTH_MAP.get(length_pref, 1) / 2.0
    vec[MAINTENANCE] = MAINTENANCE_MAP.get(maintenance, 0) / 2.0
    vec[HAIR_TYPE]   = HAIR_TYPE_MAP.get(hair_type, 0) / 3.0

    return vec


def get_recommendations(face_shape: str, gender: str,
                        hair_type: str = "any",
                        length_pref: str = "medium",
                        maintenance: str = "low",
                        top_n: int = 4) -> dict:
    """
    Returns top_n hairstyle recommendations ranked by cosine similarity.

    Args:
        face_shape  : detected face shape ('oval', 'round', etc.)
        gender      : 'male' or 'female'
        hair_type   : 'any', 'straight', 'wavy', 'curly'
        length_pref : 'short', 'medium', 'long'
        maintenance : 'low', 'medium', 'high'
        top_n       : number of recommendations to return

    Returns:
        {
          "description": "...",
          "recommended": [
            {"name": "...", "description": "...", "tip": "...", "match_score": 94.2},
            ...
          ],
          "avoid": [...]
        }
    """
    face_shape  = face_shape.lower().strip()
    gender      = gender.lower().strip()
    hair_type   = hair_type.lower().strip()
    length_pref = length_pref.lower().strip()
    maintenance = maintenance.lower().strip()

    if face_shape not in FACE_SHAPE_IDX:
        return {"error": f"Unknown face shape: {face_shape}"}
    if gender not in ("male", "female"):
        return {"error": f"Gender must be 'male' or 'female'"}

    # Build user vector
    user_vec = _build_user_vector(face_shape, gender, hair_type, length_pref, maintenance)

    # Filter hairstyles by gender
    gender_filtered = {
        name: data for name, data in HAIRSTYLES.items()
        if data["gender"] == gender
    }

    # Compute cosine similarity for each hairstyle
    scores = {}
    for name, data in gender_filtered.items():
        h_vec = data["vector"]
        sim   = cosine_similarity([user_vec], [h_vec])[0][0]
        scores[name] = round(float(sim) * 100, 1)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Build recommended list (top N)
    recommended = []
    for name, score in ranked[:top_n]:
        entry = HAIRSTYLES[name]
        
        # Build image URL 
        folder_name = "men" if entry["gender"] == "male" else "female"
        image_url = f"assets/hairstyle/{folder_name}/{name}.jpg"

        recommended.append({
            "name":        name,
            "description": entry["description"],
            "tip":         entry["tip"],
            "match_score": score,
            "image_url":   image_url
        })

    # Build avoid list — bottom 3 scoring hairstyles
    avoid = [name for name, _ in ranked[-3:]]

    # Face shape descriptions
    descriptions = {
        "oval":   "Oval faces are the most versatile — slightly longer than wide with balanced proportions.",
        "round":  "Round faces are as wide as they are long with soft curves and a rounded jaw.",
        "square": "Square faces have a strong jaw, wide forehead, and straight sides.",
        "heart":  "Heart-shaped faces have a wider forehead and narrow chin.",
        "oblong": "Oblong faces are longer than wide with a similar forehead and jaw width."
    }

    return {
        "description": descriptions[face_shape],
        "recommended": recommended,
        "avoid":       avoid
    }