from pathlib import Path
import uuid

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from lime import lime_image
from skimage.segmentation import mark_boundaries


# ==========================================
# 1. PATHS AND FLASK CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "pneumonia_model.keras"

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "static"
    / "results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Keep the current threshold for now.
# Tune it later using validation data only.
PNEUMONIA_THRESHOLD = 0.50


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please train the model first."
    )

model = tf.keras.models.load_model(str(MODEL_PATH))

print("Pneumonia detection model loaded.")
print("Model path:", MODEL_PATH)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Pneumonia threshold:", PNEUMONIA_THRESHOLD)

if model.input_shape != (None, 224, 224, 3):
    raise ValueError(
        f"Unexpected model input shape: {model.input_shape}"
    )

if model.output_shape[-1] != 1:
    raise ValueError(
        f"Expected one sigmoid output, got: {model.output_shape}"
    )

# Verify that the expected layers exist.
base_model = model.get_layer("densenet121")
pooling_layer = model.get_layer("global_average_pooling2d")
dropout_layer = model.get_layer("dropout")
classifier_layer = model.get_layer("pneumonia_probability")

# The feature layer used by Grad-CAM.
base_model.get_layer("conv5_block16_concat")


# ==========================================
# 3. IMAGE PREPARATION
# ==========================================
def prepare_image(image):
    """
    Convert an uploaded image to RGB and resize it.
    Keep pixel values in the 0-255 range because
    DenseNet preprocessing is inside the trained model.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32)

    if array.shape != (224, 224, 3):
        raise ValueError(
            f"Unexpected image shape: {array.shape}"
        )

    return np.expand_dims(array, axis=0)


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================
def predict_probabilities(images):
    """
    Return two columns:
    column 0 = NORMAL probability
    column 1 = PNEUMONIA probability

    Input pixels are raw RGB values in the 0-255 range.
    """
    images = np.asarray(images, dtype=np.float32)

    if images.ndim != 4 or images.shape[1:] != (224, 224, 3):
        raise ValueError(
            "Expected images with shape (N, 224, 224, 3)."
        )

    pneumonia_scores = model(
        images,
        training=False
    ).numpy().reshape(-1)

    # Protect against tiny numerical deviations.
    pneumonia_scores = np.clip(
        pneumonia_scores, 0.0, 1.0
    )

    normal_scores = 1.0 - pneumonia_scores

    return np.column_stack(
        [normal_scores, pneumonia_scores]
    )


# ==========================================
# 5. GENERATE GRAD-CAM
# ==========================================
def create_gradcam(batch, output_path):
    """
    Generate a Grad-CAM heatmap for the pneumonia output.
    Uses DenseNet preprocessing consistently with the model.
    """
    base = model.get_layer("densenet121")

    feature_extractor = tf.keras.Model(
        inputs=base.input,
        outputs=base.get_layer(
            "conv5_block16_concat"
        ).output
    )

    # IMPORTANT:
    # The main model preprocesses the input before DenseNet.
    # Apply the same preprocessing here.
    preprocessed = tf.keras.applications.densenet.preprocess_input(
        tf.convert_to_tensor(batch, dtype=tf.float32)
    )

    with tf.GradientTape() as tape:
        feature_maps = feature_extractor(
            preprocessed,
            training=False
        )

        tape.watch(feature_maps)

        x = pooling_layer(feature_maps)
        x = dropout_layer(x, training=False)
        pneumonia_score = classifier_layer(x)

        # Gradient target: pneumonia probability.
        score = pneumonia_score[:, 0]

    gradients = tape.gradient(score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM could not calculate gradients."
        )

    # Average gradients over spatial dimensions.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted sum of feature maps.
    heatmap = tf.reduce_sum(
        feature_maps * weights[:, None, None, :],
        axis=-1
    )[0]

    heatmap = tf.maximum(heatmap, 0.0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMG_SIZE
    ).numpy()[..., 0]

    original = batch[0].astype(np.uint8)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4)
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Chest X-ray")
    axes[0].axis("off")

    axes[1].imshow(original)
    axes[1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Grad-CAM: Pneumonia Features")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150
    )
    plt.close(fig)


# ==========================================
# 6. GENERATE LIME EXPLANATION
# ==========================================
def create_lime(batch, predicted_class, output_path):
    """
    LIME highlights image regions that influence
    the selected model class.
    """
    image = np.clip(
        batch[0], 0, 255
    ).astype(np.uint8)

    explainer = lime_image.LimeImageExplainer(
        random_state=42
    )

    explanation = explainer.explain_instance(
        image=image,
        classifier_fn=predict_probabilities,
        labels=(0, 1),
        top_labels=2,
        hide_color=0,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        label=int(predicted_class),
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    # LIME may return an image with altered pixel values.
    display_image = np.clip(
        temp, 0, 255
    ).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(
        mark_boundaries(
            display_image,
            mask
        )
    )

    ax.set_title("LIME Explanation")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150
    )
    plt.close(fig)


# ==========================================
# 7. WEBSITE ROUTE
# ==========================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("xray")

        if uploaded is None or not uploaded.filename:
            error = "Please select a chest X-ray image."

        elif Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            error = "Please upload a JPG, JPEG, or PNG image."

        else:
            try:
                # Open and validate the uploaded image.
                with Image.open(uploaded.stream) as source:
                    source.verify()

                uploaded.stream.seek(0)

                with Image.open(uploaded.stream) as source:
                    image = source.convert("RGB")

                batch = prepare_image(image)

                # Predict using the same model used by the website.
                probabilities = predict_probabilities(batch)[0]

                normal_score = float(probabilities[0])
                pneumonia_score = float(probabilities[1])

                # Apply the current decision threshold.
                if pneumonia_score >= PNEUMONIA_THRESHOLD:
                    prediction_name = "PNEUMONIA"
                    predicted_class = 1
                else:
                    prediction_name = "NORMAL"
                    predicted_class = 0

                # This is the score for the selected class,
                # not a guarantee of prediction correctness.
                confidence = float(
                    probabilities[predicted_class]
                ) * 100.0

                image_id = uuid.uuid4().hex

                gradcam_name = f"{image_id}_gradcam.png"
                lime_name = f"{image_id}_lime.png"

                gradcam_path = RESULTS_DIR / gradcam_name
                lime_path = RESULTS_DIR / lime_name

                # Generate explanations for this prediction.
                create_gradcam(
                    batch,
                    gradcam_path
                )

                create_lime(
                    batch,
                    predicted_class,
                    lime_path
                )

                result = {
                    "prediction": prediction_name,
                    "confidence": round(confidence, 2),
                    "normal_score": round(normal_score * 100, 2),
                    "pneumonia_score": round(pneumonia_score * 100, 2),
                    "gradcam": gradcam_name,
                    "lime": lime_name
                }

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                app.logger.warning(
                    "Invalid or unreadable uploaded image: %s",
                    exc
                )
                error = (
                    "This image could not be read. "
                    "Please upload a valid JPG, JPEG, or PNG image."
                )

            except Exception:
                app.logger.exception(
                    "X-ray prediction or explanation failed"
                )
                error = (
                    "Image processing failed. "
                    "Please check the VS Code terminal for details."
                )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ==========================================
# 8. START WEBSITE
# ==========================================
if __name__ == "__main__":
    # Keep debug mode disabled to avoid starting
    # the model twice through Flask's reloader.
    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )