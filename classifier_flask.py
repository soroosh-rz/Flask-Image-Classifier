from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions
from PIL import Image, ImageFile
from io import BytesIO

print(f'Tensorflow version: {tf.__version__}')

ALLOWED_EXTENSION = set(['png', 'jpg', 'jpeg', 'gif'])
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


def valid_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1] in ALLOWED_EXTENSION)


app = Flask(__name__)
model = MobileNet(weights='imagenet', include_top=True)


@app.route('/')
def index():
    return render_template('ImageML.html')


@app.route('/api/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']

    if file.filename == '':
        return render_template('ImageML.html', prediction='You did not select an image')

    if file and valid_file(file.filename):
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        x = image.img_to_array(img)

        if x.shape[2] != IMAGE_CHANNELS:
            return render_template('ImageML.html', prediction='Image color channels should be 3')

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        prd = decode_predictions(pred, top=3)  # top 3 predictions

        items = []
        for item in prd[0]:
            items.append({'name': item[1], 'prob': float(item[2])})

        response = {'predictions': items}
        return render_template('ImageML.html', prediction=response)
    else:
        return render_template('ImageML.html', prediction='Invalid File extension')


if __name__ == '__main__':
    app.run()
