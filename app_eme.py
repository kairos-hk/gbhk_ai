import os
import base64
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from tensorflow import keras
from PIL import ImageOps


model = tf.keras.models.load_model('keras_model.h5')
model_eme = keras.models.load_model('keras_model_eme.h5')

app = Flask(__name__)
CORS(app)

def preprocess_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def preprocess_image_eme(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)


def classify_image(image_data):
    preprocessed_image = preprocess_image(image_data)

    prediction = model.predict(preprocessed_image)

    category_index = tf.argmax(prediction, axis=-1).numpy()[0]
    threshold = 0.5
    if prediction.max() < threshold:
        category_index = 6
        
    return category_index

@app.route('/eme', methods=['POST'])
def process_image():
    image_data = request.json['image']
    preprocessed_image_eme = preprocess_image_eme(image_data)

    prediction = model_eme.predict(preprocessed_image_eme)
    index = np.argmax(prediction)

    if index == 0:
        result_e = {'title': '화재 긴급 신고', 'category_index': int(index)}
    elif index == 1:
        result_e = {'title': '차량 교통사고 긴급 신고', 'category_index': int(index)}
    else:
        result_e = {'title': '분류 결과 없음', 'category_index': int(index)}

    return jsonify(result_e)


@app.route('/dif', methods=['POST'])
def analyze_image():
    image_data = request.get_json()['image']

    category_index = classify_image(image_data)

    if category_index == 0:
        result = {'title': '도로, 시설물 파손 신고', 'category_index': int(category_index)}
    elif category_index == 1:
        result = {'title': '차량 및 교통 위험 신고', 'category_index': int(category_index)}
    elif category_index == 2:
        result = {'title': '대기오염 신고', 'category_index': int(category_index)}
    elif category_index == 3:
        result = {'title': '수질 오염 신고', 'category_index': int(category_index)}
    elif category_index == 4:
        result = {'title': '소방 안전 신고', 'category_index': int(category_index)}
    else:
        result = {'title': '분류 결과 없음', 'category_index': int(category_index)}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234)


