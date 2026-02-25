"""
Rule-based mapping: face shape + gender → hairstyle recommendations.
"""

HAIRSTYLE_RECOMMENDATIONS = {
    "heart": {
        "description": "Heart-shaped faces have a wider forehead and narrow chin.",
        "male": {
            "recommended": [
                {"name": "Textured Quiff", "description": "Adds volume at the front while keeping sides short, balancing the wide forehead.", "tip": "Use a matte pomade for a natural finish."},
                {"name": "Side Part Undercut", "description": "Clean sides draw attention away from the broad forehead.", "tip": "Keep the part deep and defined."},
                {"name": "Messy Fringe", "description": "Forward-swept fringe softens the forehead width.", "tip": "Avoid a perfectly straight fringe — keep it casual."},
                {"name": "Short Back and Sides with Length on Top", "description": "Volume on top balances the narrow chin.", "tip": "Ask for a taper fade on the sides."},
            ],
            "avoid": ["Slicked-back styles that expose the full forehead", "Very short buzz cuts"]
        },
        "female": {
            "recommended": [
                {"name": "Side-Swept Bangs", "description": "Softens the wide forehead and draws attention downward.", "tip": "Keep bangs long and swept to one side."},
                {"name": "Chin-Length Bob", "description": "Adds width at the jaw to balance the narrow chin.", "tip": "Works best slightly fuller at the ends."},
                {"name": "Wavy Lob", "description": "Loose waves at the chin add volume to the lower face.", "tip": "Avoid adding volume at the crown."},
                {"name": "Low Ponytail", "description": "Keeps the top flat and frames the chin beautifully.", "tip": "Let a few face-framing strands fall loose."},
            ],
            "avoid": ["Short pixie cuts with volume at crown", "High top buns", "Centre-parted sleek styles"]
        }
    },

    "oblong": {
        "description": "Oblong faces are longer than wide with a similar forehead and jaw width.",
        "male": {
            "recommended": [
                {"name": "Textured Crop", "description": "Horizontal fringe breaks up the face length.", "tip": "Ask for a French crop or textured fringe."},
                {"name": "Quiff with Volume on Sides", "description": "Width on the sides shortens the perceived face length.", "tip": "Use a round brush when blow-drying for extra volume."},
                {"name": "Curtain Bangs", "description": "Parted fringe adds width and shortens the face visually.", "tip": "Works great with medium-length hair."},
                {"name": "Buzz Cut with Beard", "description": "A full beard adds width to the jaw, balancing the length.", "tip": "Keep the beard well-groomed and shaped."},
            ],
            "avoid": ["Long straight styles", "Mohawks or styles with height on top", "Slicked-back looks"]
        },
        "female": {
            "recommended": [
                {"name": "Curtain Bangs", "description": "Shortens the face visually by breaking up the length.", "tip": "Part in the middle for the most flattering effect."},
                {"name": "Medium Layered Cut", "description": "Adds width and fullness on the sides.", "tip": "Ask for layers from the cheekbone down."},
                {"name": "Blunt Bob", "description": "Horizontal lines add perceived width and shorten length.", "tip": "Keep it at or just below the jaw."},
                {"name": "Voluminous Waves", "description": "Width-adding style that counteracts the long face shape.", "tip": "Use a large barrel curling iron for soft, wide waves."},
            ],
            "avoid": ["Very long straight hair", "Sleek centre-parted styles", "High ponytails"]
        }
    },

    "oval": {
        "description": "Oval faces are the most versatile — slightly longer than wide with balanced proportions.",
        "male": {
            "recommended": [
                {"name": "Classic Crew Cut", "description": "Clean, timeless style that shows off balanced features.", "tip": "Any variation works — go for what suits your lifestyle."},
                {"name": "Slicked Back Undercut", "description": "Highlights symmetry and gives a sharp, polished look.", "tip": "Use a strong hold pomade."},
                {"name": "Textured Pompadour", "description": "Adds personality while working with the balanced proportions.", "tip": "Keep sides faded for a modern finish."},
                {"name": "Modern Quiff", "description": "Versatile and stylish — suits oval perfectly.", "tip": "Works with straight or wavy hair."},
            ],
            "avoid": ["Almost nothing — oval is the most versatile shape!"]
        },
        "female": {
            "recommended": [
                {"name": "Classic Pixie Cut", "description": "Shows off balanced features with minimal length.", "tip": "Any variation works — textured, sleek, or with a side part."},
                {"name": "Beachy Waves", "description": "Effortless style that suits the balanced oval shape perfectly.", "tip": "Salt spray gives a natural, textured look."},
                {"name": "High Bun", "description": "Highlights facial symmetry and elongates the neck.", "tip": "Leave face-framing pieces out for a softer look."},
                {"name": "Sleek Long Hair", "description": "Clean straight hair showcases balanced proportions.", "tip": "A centre part works especially well."},
            ],
            "avoid": ["Styles that work against face symmetry — almost anything goes!"]
        }
    },

    "round": {
        "description": "Round faces are as wide as they are long, with soft curves and a rounded jaw.",
        "male": {
            "recommended": [
                {"name": "High Fade with Height on Top", "description": "Height adds length, tight sides slim the face.", "tip": "A high skin fade works best."},
                {"name": "Pompadour", "description": "Vertical volume elongates the face significantly.", "tip": "Keep sides very short to contrast the top volume."},
                {"name": "Side Part with Slick", "description": "Asymmetry breaks up the roundness.", "tip": "A deep side part is especially effective."},
                {"name": "Angular Fringe", "description": "Angled styling creates the illusion of sharper features.", "tip": "Avoid a perfectly round or bowl-shaped fringe."},
            ],
            "avoid": ["Buzz cuts with no fade", "Rounded bowl cuts", "Voluminous styles on the sides"]
        },
        "female": {
            "recommended": [
                {"name": "Long Layers", "description": "Elongates the face and draws the eye up and down.", "tip": "Ask for layers starting below the chin."},
                {"name": "High Top Knot", "description": "Adds height to the head, making the face appear longer.", "tip": "Keep sides sleek and flat."},
                {"name": "Side Part with Volume", "description": "Asymmetry breaks up the roundness and slims the face.", "tip": "A deep side part is especially effective."},
                {"name": "Straight Lob with Centre Part", "description": "Lengthens the face and creates clean vertical lines.", "tip": "Avoid flipping ends outward — keep them straight."},
            ],
            "avoid": ["Blunt bobs at jaw level", "Short styles with volume on sides", "Curly voluminous styles"]
        }
    },

    "square": {
        "description": "Square faces have a strong jaw, wide forehead, and straight sides.",
        "male": {
            "recommended": [
                {"name": "Messy Textured Top", "description": "Soft texture reduces the angular look of a square jaw.", "tip": "Avoid overly structured or hard styles."},
                {"name": "Long Fringe / Curtains", "description": "Softens the forehead and draws attention to the centre.", "tip": "Works great with medium length hair."},
                {"name": "Taper Fade with Waves", "description": "Curved natural waves soften the strong jawline.", "tip": "Let waves flow naturally — don't force structure."},
                {"name": "Longer Hair with Layers", "description": "Length past the jaw softens its angles.", "tip": "Go for at least collar-length to see the effect."},
            ],
            "avoid": ["Very short military cuts", "Hard side parts", "Blunt fringes that emphasise the forehead width"]
        },
        "female": {
            "recommended": [
                {"name": "Soft Curls or Waves", "description": "Curves soften the angular jawline.", "tip": "Medium to long length works best."},
                {"name": "Side-Swept Bangs", "description": "Diagonal lines soften the square forehead.", "tip": "Keep bangs long and angled, not blunt."},
                {"name": "Long Layered Cut", "description": "Draws attention down and away from the jaw corners.", "tip": "Face-framing layers are key."},
                {"name": "Textured Pixie", "description": "Soft texture and side-swept styling reduce the angular look.", "tip": "Avoid blunt, structured pixie cuts."},
            ],
            "avoid": ["Blunt bobs that hit at jaw", "Straight-across fringes", "Very sleek straight styles"]
        }
    }
}


def get_recommendations(face_shape: str, gender: str) -> dict:
    """
    Args:
        face_shape : one of 'heart', 'oblong', 'oval', 'round', 'square'
        gender     : 'male' or 'female'
    Returns:
        dict with description, recommended list, and avoid list
    """
    face_shape = face_shape.lower().strip()
    gender     = gender.lower().strip()

    if face_shape not in HAIRSTYLE_RECOMMENDATIONS:
        return {"error": f"Unknown face shape: {face_shape}"}

    if gender not in ("male", "female"):
        return {"error": f"Gender must be 'male' or 'female', got: {gender}"}

    entry = HAIRSTYLE_RECOMMENDATIONS[face_shape]
    return {
        "description": entry["description"],
        "recommended": entry[gender]["recommended"],
        "avoid":       entry[gender]["avoid"]
    }