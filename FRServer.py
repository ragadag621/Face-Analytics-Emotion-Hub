from flask import Flask, request, jsonify, render_template
import face_recognition
import os
from PIL import Image
from scipy.spatial.distance import cosine
from io import BytesIO
import base64
from deepface import DeepFace

app = Flask(__name__)

# Path to known faces

base_dir = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(base_dir, 'known_faces', 'images')
UPLOADS_DIR = os.path.join(base_dir, "uploads")


# Ensure the uploads directory exists
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# Load encodings for known faces
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for name in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(image_path):
        print(f"Loading image for {name}...")
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name.split(".")[0])
            print(f"Encoding for {name} loaded successfully.")
        else:
            print(f"No face found in {name}")
    else:
        print(f"Image for {name} not found at {image_path}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploadPage')
def upload_page():
    return render_template('upload.html')


@app.route('/cameraPage')
def camera_page():
    return render_template('camera.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Process the uploaded image and compare it with known faces."""
    if 'image' not in request.files:
        return jsonify({'message': 'No image uploaded'}), 400

    # Save the uploaded image
    image_file = request.files['image']
    image_path = os.path.join(UPLOADS_DIR, image_file.filename)
    image_file.save(image_path)

    return process_image(image_path)


@app.route('/capture', methods=['POST'])
def capture_image():
    """Process the image captured from the camera."""
    data = request.get_json()
    image_data = data['image']

    # Decode the sent image data
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_path = os.path.join(UPLOADS_DIR, 'captured_image.jpg')
    image.save(image_path)

    return process_image(image_path)


def process_image(image_path):
    """Process the image and compare it with known faces."""
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)

    if not unknown_encoding:
        return jsonify({'message': 'No face found in the image'}), 400

    unknown_encoding = unknown_encoding[0]

    # Variables to track the closest match
    min_distance = float('inf')
    best_match_name = None

    # Compare the encodings
    for known_encoding, name in zip(known_face_encodings, known_face_names):
        distance = cosine(known_encoding, unknown_encoding)
        print(f"Comparing with {name}, distance: {distance}")

        if distance < min_distance:
            min_distance = distance
            best_match_name = name

    THRESHOLD = 0.4  # more realistic

    # Case 1: Face not recognized
    if min_distance >= THRESHOLD:
        return jsonify({'message': "Face not recognized"}), 401

    # Case 2: Face recognized
    result = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=False
    )

    dominant_emotion = result[0]['dominant_emotion']
    emotion_scores = result[0]['emotion']

    return jsonify({
        'name': best_match_name,
        'emotion': dominant_emotion,
        'emotion_scores': emotion_scores,
        'message': f"Welcome {best_match_name}, you look {dominant_emotion} today!"
    }), 200



if __name__ == '__main__':
    app.run(debug=True)
