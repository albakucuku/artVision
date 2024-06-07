import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Update to your model's input size

# Load your trained model
model = tf.keras.models.load_model('path/to/your/model.h5')

# Load artist information from CSV
artist_info = pd.read_csv('path/to/artists.csv')

# Create a mapping from label indices to artist names
label_to_artist = {idx: artist for artist, idx in artist_to_label.items()}  # Assuming artist_to_label is defined

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        image = Image.open(filepath).resize((IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_artist = label_to_artist[predicted_label]
        
        # Get artist information
        artist_data = artist_info[artist_info['name'] == predicted_artist].iloc[0]

        return render_template(
            'result.html', 
            artist=predicted_artist, 
            filename=filename,
            years=artist_data['years'],
            genre=artist_data['genre'],
            nationality=artist_data['nationality'],
            bio=artist_data['bio'],
            wikipedia=artist_data['wikipedia'],
            paintings=artist_data['paintings']
        )
    return redirect(request.url)

# Route to display uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=os.path.join(UPLOAD_FOLDER, filename)))

if __name__ == '__main__':
    app.run(debug=True)
