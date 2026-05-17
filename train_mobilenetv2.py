import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from tensorflow.keras.models import load_model
import os  # Added to list directories

# Set paths
train_dir = r"C:\Users\Esha Shahid\Downloads\rice_leaf_dataset\train"
val_dir = r"C:\Users\Esha Shahid\Downloads\rice_leaf_dataset\validation"

# ================== EDA - Class Distribution Bar Graph ==================
categories = os.listdir(train_dir)
counts = [len(os.listdir(os.path.join(train_dir, category))) for category in categories]

plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Distribution of Rice Leaf Diseases in Training Data')
plt.xlabel('Disease Type')
plt.ylabel('Number of Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# ========================================================================

# Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=False
)

# Load Pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Add Custom Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save Model
model.save('mobilenetv2_model.keras')

# Plot Accuracy and Loss Graphs
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Load model and make predictions
model = load_model('mobilenetv2_model.keras')

# Get true labels and predicted labels
y_true = val_generator.classes
y_pred = np.argmax(model.predict(val_generator), axis=-1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


'''In this script, I begin by performing exploratory data analysis (EDA) to visualize the distribution of rice leaf disease images across 
different categories in the training dataset. I then preprocess the data using ImageDataGenerator and prepare the training and validation sets. 
The MobileNetV2 model, pre-trained on ImageNet, is loaded and customized with additional layers to classify the rice leaf diseases into three 
categories. After training the model for 10 epochs, I save the model for future use. I also generate and display graphs showing the training 
and validation accuracy and loss. Once the model is saved, I load it again to make predictions on the validation set, providing a classification
report and a confusion matrix. The confusion matrix is visualized using a heatmap to better understand the model's performance.'''