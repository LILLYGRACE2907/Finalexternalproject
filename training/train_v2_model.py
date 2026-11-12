import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

sys.stdout.reconfigure(line_buffering=True)

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
LABEL_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_split_dataset(split_name):
    """Load 48x48 images and labels into compact uint8 memory arrays."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_dir = os.path.join(base_dir, "dataset_split", split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Directory {split_dir} not found.")

    images = []
    labels = []
    print(f"Loading {split_name} split from {split_dir}...")
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            fpath = os.path.join(class_folder, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (48, 48):
                images.append(img)
                labels.append(class_idx)

    X = np.array(images, dtype=np.uint8)[..., np.newaxis]
    y = np.array(labels, dtype=np.int32)
    print(f"[+] Loaded {len(X)} {split_name} samples ({X.nbytes / 1024 / 1024:.2f} MB)")
    return X, y

def build_emotion_cnn():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),

        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),

        # Dense Head with GAP to prevent overfitting & keep CPU compute fast
        layers.GlobalAveragePooling2D(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Dense(7, activation='softmax')
    ])
    return model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    v2_model_path = os.path.join(models_dir, "emotion_model_v2.keras")

    print("=" * 70)
    print("TRAINING ENHANCED ALL-7 EMOTION RECOGNITION CNN (v2)")
    print("=" * 70)

    # 1. Load train and val splits
    X_train_u8, y_train = load_split_dataset("train")
    X_val_u8, y_val = load_split_dataset("val")

    # Normalize to [0, 1] float32
    X_train = X_train_u8.astype(np.float32) / 255.0
    X_val = X_val_u8.astype(np.float32) / 255.0

    # 2. Smooth class weights: (N_max / N_c)^0.35
    # Moderate weighting incentivizes minority learning without destabilizing gradients
    counts = np.bincount(y_train, minlength=7)
    max_count = np.max(counts)
    class_weights = {i: float((max_count / counts[i]) ** 0.35) for i in range(7)}
    print("\nSmoothed Class Weights:")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10}: count={counts[idx]:<5} | weight={class_weights[idx]:.4f}")

    # 3. Build model
    model = build_emotion_cnn()
    print(f"\nModel initialized with {model.count_params():,} parameters.")

    # 4. Compile with Label Smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # 5. Fast tf.data pipeline with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=4096, seed=42)
    train_ds = train_ds.map(
        lambda img, lbl: (tf.image.random_flip_left_right(img), lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(64).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    # 6. Callbacks: checkpoint best val_accuracy, reduce LR on plateau, early stopping
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=v2_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\nStarting model training for 12 epochs...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1
    )
    total_time = time.time() - t0
    print(f"\n[+] Training completed in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"[+] Best validated model saved to: {v2_model_path}")

if __name__ == "__main__":
    main()
