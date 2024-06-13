import os
os.environ['TFCPPMINLOGLEVEL'] = '2'

from flask import Flask, request, jsonify 
import tensorflow as tf 
import io
from tensorflow import keras
import numpy as np
from PIL import Image
import json

# Load model h.5 pake keras
model = keras.models.load_model('mobilenet3.h5')
with open('class_descriptions.json', 'r') as f:
    class_names = json.load(f)

app = Flask(__name__)

#Function untuk melakukan prediksi pada gambar yang di input
def predict_label(img):
    print("test")
    # image = img.resize((224, 224))
    i = np.asarray(img)/255.0
    i = i.reshape(1,224, 224,3)
    pred = model.predict(i)
    predicted_class = np.argmax(pred)
    if predicted_class < len(class_names):
        result = class_names[predicted_class]
    else:
        result = "Unknown label"
    return result


@app.route('/predict', methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224), Image.NEAREST)
        pred_img = predict_label(img)
        print("start predict")
        return pred_img
    except Exception as e:
        return jsonify({"error predict": str(e)})
    finally:
        if file: 
            file.close()


# @app.route('/predict', methods=["GET", "POST"])
# def index(): 
#     file = request.files.get('file')
#     if file is None or file.filename == "":
#         return jsonify({"error": "no file"})
#     try:
#         result = predict_label(file)
#         return jsonify({"result": result})
#     except Exception as e:
#         return jsonify({"error": e})


if __name__ == '__main__':
    app.run(debug=True)