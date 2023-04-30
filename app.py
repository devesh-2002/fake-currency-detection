import numpy as np
from flask import Flask, request, render_template,redirect,url_for
import cv2
import tensorflow as tf
from keras.models import load_model
import os 
from PIL import Image
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path='/static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('vgg_model_2.h5')

def process_jpg_image(img):
    img = tf.convert_to_tensor(img[:,:,:3])
    img = np.expand_dims(img, axis = 0)
    img = tf.image.resize(img,[224,224])
    img = (img/255.0)
    return img

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        print(file_path)
        
        file.save(file_path)
        file_path = os.path.join('uploads', file.filename)
        test_image_read_1 = cv2.imread(file_path)
        test_image_1 = process_jpg_image(test_image_read_1)
        print(test_image_1)
        prediction_1 = model.predict(test_image_1)
        print(prediction_1)
        return render_template('index.html', prediction_text=prediction_1, image_file_name=file.filename)

if __name__ == "__main__":
    app.run(debug=True)