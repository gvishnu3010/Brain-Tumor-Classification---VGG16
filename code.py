from google.colab import drive
drive.mount('/content/drive')

import zipfile

zip_path = "/content/drive/MyDrive/datasets/archive.zip"
extract_path = "/content/data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

!ls /content/data

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

#DATA PREPROCESSING
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen_cnn = train_datagen.flow_from_directory(
    "/content/data/Training",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen_cnn = val_datagen.flow_from_directory(
    "/content/data/Testing",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


print(train_gen_cnn.class_indices)
print("Training images:", train_gen_cnn.samples)
print("Validation images:", val_gen_cnn.samples)


import matplotlib.pyplot as plt
import numpy as np

# Get one batch of images and labels
images, labels = next(train_gen_cnn)

plt.figure(figsize=(6,6))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title("Label index: " + str(np.argmax(labels[i])))
    plt.axis("off")

plt.show()


!ls /content/data/Training


classes = list(train_gen_cnn.class_indices.keys())

{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

print(train_gen_cnn.class_indices)

classes = list(train_gen_cnn.class_indices.keys())
print("Classes:", classes)

#DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

#MODEL BUILDING
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(len(classes), activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

#COMPIE MODEL
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#TRAIN MODEL
history = model.fit(
    train_gen_cnn,
    validation_data=val_gen_cnn,
    epochs=15
)

#PREDICTION FUNCTION - “The prediction function takes a new MRI image, preprocesses it, and passes it to the trained model to classify the tumor type.”
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return classes[np.argmax(pred)]

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Reset the validation generator to ensure predictions are made on the same order as true labels
val_gen_cnn.reset()

# Get true labels
y_true = val_gen_cnn.classes

# Get predictions
predictions = model.predict(val_gen_cnn)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=val_gen_cnn.class_indices.keys())

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

#GRAD_CAM
def get_gradcam(img_array, model):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("block5_conv3").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

#HEAT-MAP:
import cv2
import numpy as np

classes = list(train_gen_cnn.class_indices.keys())

# Define a sample image path for demonstration
# You can change this to any image path you want to test
img_path = "/content/data/Testing/glioma/Te-gl_0010.jpg"

# Load and preprocess the image
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Image not found at {img_path}")
else:
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    pred = model.predict(img_array)[0]

    # sort predictions
    result = dict(sorted(
        {classes[i]: float(pred[i]) for i in range(len(classes))}.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    print("Prediction results:", result)

def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + img
    return superimposed

#TESTING THE MODEL
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define classes using the order from train_gen_cnn.class_indices
# This assumes train_gen_cnn has been defined and executed previously.
classes = list(train_gen_cnn.class_indices.keys())

# 1. Select an image from dataset
img_path = "/content/data/Testing/meningioma/Te-me_0018.jpg"   # Corrected image name

# 2. Prediction
result = predict_image(img_path)
print("Prediction:", result)

# 3. Load image
img = cv2.imread(img_path)

if img is None:
    print("Image not found")
else:
    # 4. Preprocess image
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 5. Generate Grad-CAM heatmap
    heatmap = get_gradcam(img, model)

    # 6. Overlay heatmap on original image
    output = overlay_heatmap(img_path, heatmap)

    # 7. Show result
    plt.imshow(output.astype("uint8"))
    plt.axis("off")
    plt.show()

import matplotlib.pyplot as plt

# Example: use your model history
# history = model.fit(...)

# Training accuracy
train_acc = history.history['accuracy']

# Validation accuracy
val_acc = history.history['val_accuracy']

# Epochs
epochs = range(1, len(train_acc) + 1)

# Plot graph
plt.figure()
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

import matplotlib.pyplot as plt

models = ['VGG16 (Basic)', 'VGG16 (Augmented)']
accuracy = [0.92, 0.96]   # replace with your values

plt.figure()
plt.bar(models, accuracy)

plt.title('Performance Improvement')
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.show()

from sklearn.metrics import classification_report
import numpy as np

# Reset the validation generator to ensure predictions are made on the same order as true labels
val_gen_cnn.reset()

# Get true labels from the validation generator
y_true = val_gen_cnn.classes

# Get predictions from the model on the validation data
predictions = model.predict(val_gen_cnn)
y_pred_classes = np.argmax(predictions, axis=1)

# Class names (using the order from val_gen_cnn.class_indices)
class_names = list(val_gen_cnn.class_indices.keys())

# Generate classification report
report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)

# Print formatted table
print("--- ADVANCED EVALUATION METRICS ---")
print(f"{'Tumor Type':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support'}")

for cls in class_names:
    precision = report[cls]['precision']
    recall = report[cls]['recall']
    f1 = report[cls]['f1-score']
    support = int(report[cls]['support'])

    print(f"{cls:<15}{precision:<12.5f}{recall:<12.5f}{f1:<12.5f}{support}")

import matplotlib.pyplot as plt

# Tumor classes
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Your actual values
precision = [0.83108, 0.80769, 0.88435, 0.87059]
recall    = [0.82000, 0.61765, 0.96296, 0.98667]
f1_score  = [0.82550, 0.70000, 0.92199, 0.92500]

plt.figure()

plt.plot(classes, precision, marker='o', label='Precision')
plt.plot(classes, recall, marker='o', label='Recall')
plt.plot(classes, f1_score, marker='o', label='F1-score')

plt.title('Performance Comparison (Line Graph)')
plt.xlabel('Tumor Classes')
plt.ylabel('Score')
plt.legend()

plt.show()

#GRADIO:
import gradio as gr
import cv2
import numpy as np
from PIL import Image

# Class names (same as training)
classes = list(train_gen_cnn.class_indices.keys())

# This uses your existing predict_image() logic
def gradio_predict(img):

    # Convert image (Gradio gives numpy array)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    pred = model.predict(img_array)

    # Convert to readable output
    result = {classes[i]: float(pred[0][i]) for i in range(4)}

    return result


# Create interface
interface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=4),
    title="Brain Tumor Classification (VGG16)",
    description="Upload an MRI image to classify tumor type (Glioma, Meningioma, Pituitary, Normal)"
)

# Launch app
interface.launch(debug=True)
