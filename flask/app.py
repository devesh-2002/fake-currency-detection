import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import cv2
import tensorflow as tf
from keras.models import load_model
import os 
from PIL import Image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path='/static')
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('models/vgg_model_2.h5')

def process_jpg_image(img):
    img = tf.convert_to_tensor(img[:,:,:3])
    img = np.expand_dims(img, axis=0)
    img = tf.image.resize(img, [224, 224])
    img = (img / 255.0)
    return img

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Fake Currency Detection API'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        file_path = os.path.join('uploads', file.filename)
        test_image_read_1 = cv2.imread(file_path)
        test_image_1 = process_jpg_image(test_image_read_1)
        prediction_1 = model.predict(test_image_1)
        print(prediction_1[0][0], prediction_1[0][1])
        return jsonify({'prediction_text': prediction_1.tolist(), 'image_file_name': file.filename})

if __name__ == "__main__":
    app.run(debug=True)
