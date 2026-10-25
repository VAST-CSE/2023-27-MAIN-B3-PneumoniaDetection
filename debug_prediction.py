from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model(
    "model/pneumonia_model.keras"
)

# Select one Normal test image
normal_folder = Path("dataset/test/NORMAL")
image_path = next(
    f for f in normal_folder.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

# Prepare image exactly as the website does
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

image_array = np.asarray(
    image, dtype=np.float32
)[np.newaxis, ...]

# Predict without additional preprocessing
prediction = model.predict(
    image_array, verbose=0
)[0][0]

print("Image:", image_path.name)
print("Raw model output:", float(prediction))
print("Normal score:", round((1 - prediction) * 100, 2), "%")
print("Pneumonia score:", round(prediction * 100, 2), "%")
print(
    "Predicted class:",
    "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
)