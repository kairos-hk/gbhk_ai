import os
import base64
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

model = tf.keras.models.load_model('keras_model.h5')

app = Flask(__name__)
CORS(app)

def preprocess_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def classify_image(image_data):
    preprocessed_image = preprocess_image(image_data)

    prediction = model.predict(preprocessed_image)

    category_index = tf.argmax(prediction, axis=-1).numpy()[0]
    threshold = 0.5
    if prediction.max() < threshold:
        category_index = 6
        
    return category_index

@app.route('/eme', methods=['POST'])
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
