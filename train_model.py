import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Set paths
train_dir = r"C:\Users\Esha Shahid\Downloads\rice_leaf_dataset\train"
val_dir = r"C:\Users\Esha Shahid\Downloads\rice_leaf_dataset\validation"

# Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=False
)

# Function to build, train and evaluate a model
def train_and_evaluate(base_model_fn, model_name):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=10, validation_data=val_generator)

    # Predict and evaluate
    y_true = val_generator.classes
    y_pred = np.argmax(model.predict(val_generator), axis=-1)

    report = classification_report(y_true, y_pred, output_dict=True)

    accuracy = history.history['val_accuracy'][-1]
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return accuracy, precision, recall, f1

# Train all models
mobilenetv2_metrics = train_and_evaluate(MobileNetV2, "MobileNetV2")
vgg16_metrics = train_and_evaluate(VGG16, "VGG16")
resnet50_metrics = train_and_evaluate(ResNet50, "ResNet50")

# Plot bar graph for Accuracy
models_names = ['MobileNetV2', 'VGG16', 'ResNet50']
accuracies = [mobilenetv2_metrics[0], vgg16_metrics[0], resnet50_metrics[0]]

plt.figure(figsize=(8, 6))
plt.bar(models_names, accuracies, color=['blue', 'green', 'red'])
plt.title('Model Comparison: Validation Accuracy')
plt.xlabel('Models')
plt.ylabel('Validation Accuracy')
plt.ylim(0, 1)
plt.show()



'''In this script, I am training and comparing three different CNN models — MobileNetV2, VGG16, and ResNet50 
for the task of rice leaf disease classification. The images are loaded from the training and validation folders and preprocessed using 
ImageDataGenerator. Each model's base layers are frozen, and I add custom layers on top for classification. After training, I evaluate the 
models based on accuracy, precision, recall, and F1-score. Finally, I plot a bar graph to visually compare the validation accuracies 
of each model.'''