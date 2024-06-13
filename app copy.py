
from flask import Flask, request
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import json
import io

app = Flask(__name__)

# Load your pre-trained model
model = load_model('mobilenet3.h5')

# Load class labels
with open('lables.json', 'r') as file:
    class_labels = json.load(file)
class_labels = {int(k): v for k, v in class_labels.items()}


def load_image(file, target_size=(224, 224)):

# Open the file stream
    file_stream = io.BytesIO(file.stream.read())
    
    # Load the image from the file stream
    img = load_img(file_stream, target_size=target_size)
    
    # Reset the file stream position
    file_stream.seek(0)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        img_array = load_image(file, target_size=(224, 224))
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        result = {
            'prediction': str(predicted_label),
            'probability': str(predictions[0].tolist())
        }
        return result
    else:
        return 'No file uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)
