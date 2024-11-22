from flask import Flask, render_template, request, jsonify
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
TF_ENABLE_ONEDNN_OPTS=0
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img.save('static/file.jpg')

    # Read the saved image and detect face
    img1 = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')  # Load the Haar Cascade for face detection
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    # Draw rectangle around detected face and crop it
    for x, y, w, h in faces:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = img1[y:y + h, x:x + w]

    # Save the processed images
    cv2.imwrite('static/after.jpg', img1)
    try:
        cv2.imwrite('static/cropped.jpg', cropped)
    except:
        pass

    # Load and preprocess the cropped image
    try:
        image = cv2.imread('static/cropped.jpg', 0)  # Load the cropped face image
    except:
        image = cv2.imread('static/file.jpg', 0)  # Use the original image if no face detected

    image = cv2.resize(image, (48, 48))  # Resize to model's input size
    image = image / 255.0  # Normalize the pixel values
    image = np.reshape(image, (1, 48, 48, 1))  # Reshape to model input format

    # Load your pre-trained emotion detection model
    model = load_model('model.h5')

    # Predict the emotion
    prediction = model.predict(image)
    label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']  # Class labels
    prediction = np.argmax(prediction)  # Get index of the predicted label
    final_prediction = label_map[prediction]

    # Return the predicted emotion as JSON
    return jsonify({'prediction': final_prediction})


if __name__ == "__main__":
    app.run(port=3000, debug=True)
