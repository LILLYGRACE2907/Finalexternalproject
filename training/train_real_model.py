
from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset_split"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_dir = DATASET / "train"
val_dir = DATASET / "val"

if not train_dir.exists() or not val_dir.exists():
    raise SystemExit("ERROR: dataset_split/train or val folder not found.")

BATCH_SIZE = 4
IMG_SIZE = (48, 48)
SEED = 42
EPOCHS = 15

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names
print("Emotion class order:", class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

if class_names != val_ds.class_names:
    raise SystemExit("ERROR: Training and validation classes do not match.")

if len(class_names) != 7:
    raise SystemExit("ERROR: Expected 7 emotion classes.")

def normalize(images, labels):
    return tf.cast(images, tf.float32) / 255.0, labels

# No cache, no AUTOTUNE, and no prefetch to reduce memory use.
train_ds = train_ds.map(normalize, num_parallel_calls=1)
val_ds = val_ds.map(normalize, num_parallel_calls=1)

labels_path = MODEL_DIR / "emotion_labels_trained.txt"
labels_path.write_text(
    "\n".join(name.capitalize() for name in class_names) + "\n",
    encoding="utf-8"
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

output_path = MODEL_DIR / "emotion_model_trained.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(output_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

model.summary()

print("\nStarting low-memory training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining finished.")
print("Model path:", output_path)
print("Labels path:", labels_path)
