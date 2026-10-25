import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ------------------------------------------
# 1. Settings
# ------------------------------------------
IMG_SIZE = (224, 224)

# ------------------------------------------
# 2. Load the trained model
# ------------------------------------------
model = tf.keras.models.load_model(
    "model/pneumonia_model.keras"
)

print("Trained model loaded successfully!")

# ------------------------------------------
# 3. Load a test X-ray image
# ------------------------------------------
image_folder = Path("dataset/test/PNEUMONIA")

image_files = [
    file for file in image_folder.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

if not image_files:
    raise FileNotFoundError(
        "No X-ray images found in dataset/test/PNEUMONIA"
    )

image_path = image_files[0]

image = Image.open(image_path).convert("RGB")
image = image.resize(IMG_SIZE)

image_array = np.array(image, dtype=np.float32)
batch = np.expand_dims(image_array, axis=0)

print("X-ray image loaded:", image_path.name)

# ------------------------------------------
# 4. Get the DenseNet121 feature extractor
# ------------------------------------------
base_model = model.get_layer("densenet121")

feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer(
        "conv5_block16_concat"
    ).output
)

# ------------------------------------------
# 5. Apply preprocessing
# ------------------------------------------
processed_batch = (
    tf.keras.applications.densenet.preprocess_input(
        batch.copy()
    )
)

# ------------------------------------------
# 6. Calculate Grad-CAM heatmap
# ------------------------------------------
with tf.GradientTape() as tape:

    feature_maps = feature_extractor(
        processed_batch,
        training=False
    )

    tape.watch(feature_maps)

    # Pass feature maps through the trained classifier
    x = model.get_layer("global_average_pooling2d")(
        feature_maps
    )

    x = model.get_layer("dropout")(
        x,
        training=False
    )

    prediction = model.get_layer("dense")(x)

    score = prediction[:, 0]

# Calculate gradients
grads = tape.gradient(score, feature_maps)

if grads is None:
    raise RuntimeError(
        "Gradients could not be calculated. "
        "Check the model architecture."
    )

# Calculate the importance of each feature map
weights = tf.reduce_mean(
    grads,
    axis=(1, 2)
)

# Combine feature maps
heatmap = tf.reduce_sum(
    feature_maps * weights[:, None, None, :],
    axis=-1
)[0]

# Keep positive contributions
heatmap = tf.maximum(heatmap, 0)

# Normalize the heatmap
heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)

# ------------------------------------------
# 7. Display the original image and heatmap
# ------------------------------------------
heatmap_resized = tf.image.resize(
    heatmap[..., tf.newaxis],
    IMG_SIZE
)[..., 0].numpy()

plt.figure(figsize=(12, 5))

# Original X-ray
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Chest X-ray")
plt.axis("off")

# Grad-CAM explanation
plt.subplot(1, 2, 2)
plt.imshow(image, cmap="gray")
plt.imshow(
    heatmap_resized,
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    "model/gradcam_result.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

# ------------------------------------------
# 8. Print prediction information
# ------------------------------------------
pneumonia_score = float(prediction[0, 0])

print("\n----------- RESULTS -----------")
print("Image:", image_path.name)
print(
    "Model pneumonia score:",
    round(pneumonia_score * 100, 2), "%"
)
print(
    "Predicted class:",
    "PNEUMONIA" if pneumonia_score >= 0.5
    else "NORMAL"
)
print("Grad-CAM image saved to model/gradcam_result.png")
print("Grad-CAM completed successfully!")