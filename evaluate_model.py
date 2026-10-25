import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model = tf.keras.models.load_model(
    "model/pneumonia_model.keras"
)

# Load test images WITHOUT extra preprocessing
test_ds = tf.keras.utils.image_dataset_from_directory(
    Path("dataset") / "test",
    image_size=(224, 224),
    batch_size=16,
    label_mode="binary",
    shuffle=False
)

# Get actual labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_ds
]).astype(int)

# Model handles preprocessing internally
probabilities = model.predict(test_ds, verbose=1).ravel()
y_pred = (probabilities >= 0.5).astype(int)

print("\nClass order:", test_ds.class_names)
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_ds.class_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Check Normal images specifically
normal_indices = np.where(y_true == 0)[0]
normal_predictions = y_pred[normal_indices]

print("\nNormal images:", len(normal_indices))
print(
    "Normal correctly identified:",
    int(np.sum(normal_predictions == 0))
)
print(
    "Normal incorrectly classified as pneumonia:",
    int(np.sum(normal_predictions == 1))
)