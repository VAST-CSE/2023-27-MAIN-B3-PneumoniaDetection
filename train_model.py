from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# 1. SETTINGS
# ==========================================
DATA_DIR = Path("dataset") / "train"
MODEL_DIR = Path("model")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 8

AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Preparing the training dataset...")


# ==========================================
# 2. COLLECT IMAGE PATHS AND LABELS
# NORMAL = 0, PNEUMONIA = 1
# ==========================================
class_names = ["NORMAL", "PNEUMONIA"]

image_paths = []
labels = []

valid_extensions = {".jpg", ".jpeg", ".png"}

for label, class_name in enumerate(class_names):
    folder = DATA_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder.resolve()}"
        )

    files = sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    )

    image_paths.extend(str(path) for path in files)
    labels.extend([label] * len(files))

image_paths = np.array(image_paths)
labels = np.array(labels, dtype=np.int32)

if len(image_paths) == 0:
    raise ValueError(
        f"No JPG, JPEG or PNG images found in {DATA_DIR.resolve()}"
    )

if len(np.unique(labels)) != 2:
    raise ValueError(
        "Both NORMAL and PNEUMONIA folders must contain images."
    )

print("Total images:", len(image_paths))
print("Normal images:", int(np.sum(labels == 0)))
print("Pneumonia images:", int(np.sum(labels == 1)))


# ==========================================
# 3. STRATIFIED TRAIN / VALIDATION SPLIT
# ==========================================
train_paths, val_paths, y_train, y_val = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

print("\nTraining images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Validation Normal:", int(np.sum(y_val == 0)))
print("Validation Pneumonia:", int(np.sum(y_val == 1)))


# ==========================================
# 4. LOAD AND RESIZE IMAGES
# Pixels remain in the 0-255 range.
# DenseNet preprocessing is inside the model.
# ==========================================
def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    # Each image has shape (224, 224, 3).
    image = tf.ensure_shape(
        image,
        [224, 224, 3]
    )

    return image, label


# ==========================================
# 5. BUILD TRAINING DATASET
# Augmentation will happen inside the model ONLY.
# ==========================================
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, y_train)
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)


# ==========================================
# 6. BUILD VALIDATION DATASET
# No augmentation on validation images.
# ==========================================
val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, y_val)
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ==========================================
# 7. CLASS WEIGHTS
# Helps address the class imbalance.
# ==========================================
classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# ==========================================
# 8. DATA AUGMENTATION LAYER
# Runs only when model training=True.
# ==========================================
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal",
            seed=SEED
        ),
        tf.keras.layers.RandomRotation(
            0.03,
            seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            0.05,
            seed=SEED
        ),
    ],
    name="data_augmentation"
)


# ==========================================
# 9. BUILD DENSENET121 MODEL
# ==========================================
base_model = tf.keras.applications.DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="chest_xray"
)

# Augmentation is applied once, inside the model.
x = augmentation(inputs)

# Keep DenseNet preprocessing inside the model.
x = tf.keras.applications.densenet.preprocess_input(x)

# Initially, DenseNet weights are frozen.
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling2d"
)(x)

x = tf.keras.layers.Dropout(
    0.4,
    name="dropout"
)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="pneumonia_probability"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

print("\nModel input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

if model.input_shape != (None, 224, 224, 3):
    raise ValueError(
        f"Unexpected model input shape: {model.input_shape}"
    )


# ==========================================
# 10. COMPILE MODEL
# ==========================================
def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )


# ==========================================
# 11. CHECKPOINT AND CALLBACKS
# ==========================================
best_path = MODEL_DIR / "best_model.keras"
final_path = MODEL_DIR / "pneumonia_model.keras"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(best_path),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)


def make_callbacks():
    # Create fresh EarlyStopping and ReduceLROnPlateau
    # instances for each training phase.
    return [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            mode="min",
            verbose=1
        )
    ]


# ==========================================
# 12. PHASE ONE: TRAIN CLASSIFICATION HEAD
# ==========================================
print("\nPHASE 1: Training the classification head")

compile_model(1e-4)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(),
    verbose=1
)


# ==========================================
# 13. PHASE TWO: FINE-TUNE DENSENET121
# ==========================================
print("\nPHASE 2: Fine-tuning DenseNet121")

base_model.trainable = True

# Freeze earlier layers.
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in base_model.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

compile_model(1e-5)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(),
    verbose=1
)


# ==========================================
# 14. LOAD BEST VALIDATION CHECKPOINT
# ==========================================
print("\nLoading the best validation checkpoint...")

if not best_path.exists():
    raise FileNotFoundError(
        "Best checkpoint was not created. "
        "Check the training output for errors."
    )

best_model = tf.keras.models.load_model(
    str(best_path)
)

# Save the best checkpoint as the website model.
best_model.save(str(final_path))

with open(
    MODEL_DIR / "class_names.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(class_names, file, indent=4)

print("\nTraining completed successfully!")
print("Best checkpoint:", best_path.resolve())
print("Website model:", final_path.resolve())
print("Class order:", class_names)
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("The dataset/test folder was not used for training.")
print(
    "\nNext step: run python evaluate_model.py "
    "to evaluate on the separate test dataset."
)