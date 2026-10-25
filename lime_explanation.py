import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Settings
IMG_SIZE = (224, 224)

# Load trained model
model = tf.keras.models.load_model(
    "model/pneumonia_model.keras"
)
print("Model loaded successfully!")

# Find a test X-ray
image_folder = Path("dataset/test/PNEUMONIA")
image_path = next(
    file for file in image_folder.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

image = Image.open(image_path).convert("RGB")
image = image.resize(IMG_SIZE)
image_array = np.array(image, dtype=np.uint8)

# LIME requires a function returning class probabilities
def predict_images(images):
    images = np.asarray(images, dtype=np.float32)

    processed = (
        tf.keras.applications.densenet.preprocess_input(
            images.copy()
        )
    )

    pneumonia_prob = model.predict(
        processed, verbose=0
    ).reshape(-1)

    # Class order: NORMAL, PNEUMONIA
    normal_prob = 1.0 - pneumonia_prob

    return np.column_stack([
        normal_prob,
        pneumonia_prob
    ])

# Generate LIME explanation
explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    image_array,
    predict_images,
    top_labels=2,
    hide_color=0,
    num_samples=500
)

# Show regions supporting the predicted class
probabilities = predict_images(
    image_array[np.newaxis, ...]
)[0]

predicted_class = int(np.argmax(probabilities))

temp, mask = explanation.get_image_and_mask(
    predicted_class,
    positive_only=True,
    num_features=8,
    hide_rest=False
)

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title("Original Chest X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title("LIME Explanation")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    "model/lime_result.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

print("\nPrediction:", ["NORMAL", "PNEUMONIA"][predicted_class])
print("Normal score:", round(float(probabilities[0]) * 100, 2), "%")
print("Pneumonia score:", round(float(probabilities[1]) * 100, 2), "%")
print("LIME explanation saved to model/lime_result.png")