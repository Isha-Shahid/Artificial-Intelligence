from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model
model = tf.keras.models.load_model('mobilenetv2_model.keras')

# Class names & suggestions
class_names = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'Leaf_Smut']
suggestions = {
    'Bacterial_Leaf_Blight': 'Avoid overhead watering, use copper-based fungicides, and ensure good air circulation.',
    'Brown_Spot': 'Remove infected leaves, avoid wetting leaves while watering, and use recommended fungicides.',
    'Leaf_Smut': 'Use resistant rice varieties, ensure proper field drainage, and apply fungicides early.'
}

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/', methods=['GET', 'POST'])
def get_name():
    if request.method == 'POST':
        username = request.form['username'].capitalize()  # Capitalize the first letter
        session['username'] = username
        return redirect(url_for('upload_image'))
    session.clear()
    return render_template('name.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    username = session.get('username')
    if not username:
        return redirect(url_for('get_name'))

    error_message = None
    prediction = None
    suggestion = None
    img_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error_message = "Please select an image to upload."
        else:
            ext = file.filename.rsplit('.', 1)[-1].lower()
            if ext not in ('jpg', 'jpeg', 'png'):
                error_message = "Invalid file type — only JPG, JPEG or PNG allowed."
            else:
                # Save and predict
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(img_path)

                img_array = preprocess_image(img_path)
                preds = model.predict(img_array)
                prediction = class_names[np.argmax(preds)]
                suggestion = suggestions.get(prediction, "")

    return render_template(
        'upload.html',
        username=username,
        error_message=error_message,
        prediction=prediction,
        suggestion=suggestion,
        img_path=img_path
    )

if __name__ == '__main__':
    app.run(debug=True)


'''In this script, I am training and comparing three different CNN models — MobileNetV2, VGG16, and ResNet50 
for the task of rice leaf disease classification. The images are loaded from the training and validation folders and preprocessed using 
ImageDataGenerator. Each model's base layers are frozen, and I add custom layers on top for classification. After training, I evaluate the 
models based on accuracy, precision, recall, and F1-score. Finally, I plot a bar graph to visually compare the validation accuracies 
of each model.'''