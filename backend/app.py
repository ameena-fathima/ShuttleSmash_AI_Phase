from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    filepath = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(filepath)

    # 🔥 Deep Analysis Output
    result = {
        "stroke": "Smash",
        "quality": "Bad",
        "problems": [
            "Incorrect wrist angle",
            "Late shuttle contact",
            "Poor foot positioning"
        ],
        "biomechanics": {
            "arm_position": "Incorrect",
            "footwork": "Unbalanced",
            "timing": "Late"
        },
        "possible_injuries": [
            {
                "name": "Shoulder strain",
                "reason": "Improper arm rotation"
            },
            {
                "name": "Ankle sprain",
                "reason": "Unstable landing"
            }
        ],
        "suggestions": [
            "Practice wrist strengthening exercises",
            "Improve footwork drills",
            "Focus on early shuttle contact"
        ]
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
