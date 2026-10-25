from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

DATA_DIR = Path("dataset/train")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

model = tf.keras.models.load_model(
    "model/pneumonia_model.keras"
)

# Collect image paths and labels explicitly.
class_names = ["NORMAL", "PNEUMONIA"]
paths = []
labels = []

for label, class_name in enumerate(class_names):
    folder = DATA_DIR / class_name

    for path in folder.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(path))
            labels.append(label)

paths = np.array(paths)
labels = np.array(labels)

# Create a stratified validation split, keeping both classes.
_, val_paths, _, y_true = train_test_split(
    paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print("Validation class counts:")
print("NORMAL:", int(np.sum(y_true == 0)))
print("PNEUMONIA:", int(np.sum(y_true == 1)))

# Load images as RGB pixels (0-255).
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, y_true)
)

val_ds = (
    val_ds
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# The saved model contains DenseNet preprocessing.
# Do not preprocess the images again here.
probabilities = model.predict(
    val_ds.map(lambda x, y: x),
    verbose=1
).reshape(-1)

print("\nThreshold comparison:")
print("Threshold | Accuracy | Balanced accuracy | FP | FN")

results = []

for threshold in np.arange(0.30, 0.71, 0.05):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp)
    sensitivity_normal = tn / max(tn + fp, 1)
    sensitivity_pneumonia = tp / max(tp + fn, 1)
    balanced = (
        sensitivity_normal + sensitivity_pneumonia
    ) / 2

    results.append((
        balanced, -fn, threshold, predictions
    ))

    print(
        f"{threshold:.2f}      | {accuracy:.3f}    | "
        f"{balanced:.3f}             | {fp} | {fn}"
    )

best = max(results, key=lambda row: (row[0], row[1]))
best_threshold = best[2]
best_predictions = best[3]

print("\nSelected validation threshold:", best_threshold)
print("\nConfusion matrix:")
print(confusion_matrix(y_true, best_predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    best_predictions,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0
))