import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
LABEL_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_split_dataset(split_name):
    """Load 48x48 images and labels into compact uint8 memory arrays."""
    split_dir = os.path.join("dataset_split", split_name)
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

def compute_balanced_weights(y):
    """Compute balanced class weights: n_samples / (n_classes * class_count)."""
    classes = np.unique(y)
    n_samples = len(y)
    n_classes = len(classes)
    weights = {}
    for c in classes:
        count = int(np.sum(y == c))
        weights[int(c)] = float(n_samples / (n_classes * count))
    return weights

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    original_model_path = os.path.join(models_dir, "emotion_model.h5")
    improved_model_path = os.path.join(models_dir, "emotion_model_improved.keras")

    if not os.path.isfile(original_model_path):
        raise FileNotFoundError(f"Original model not found at {original_model_path}")

    print("=" * 70)
    print("EMOTION MODEL FINE-TUNING & CLASS-BALANCE ENHANCEMENT")
    print("=" * 70)

    # 1. Load train and val splits
    X_train_u8, y_train = load_split_dataset("train")
    X_val_u8, y_val = load_split_dataset("val")

    # Normalize to [0, 1] float32
    X_train = X_train_u8.astype(np.float32) / 255.0
    X_val = X_val_u8.astype(np.float32) / 255.0

    # 2. Compute balanced class weights to solve minority class blindness (especially Disgust & Fear)
    class_weights = compute_balanced_weights(y_train)
    print("\nComputed Class Weights:")
    for idx, cls_name in enumerate(LABEL_NAMES):
        count = int(np.sum(y_train == idx))
        w = class_weights[idx]
        print(f"  {cls_name:<10}: count={count:<5} | weight={w:.4f}")

    # 3. Load base model architecture and weights
    print(f"\nLoading baseline model: {original_model_path}")
    model = tf.keras.models.load_model(original_model_path, compile=False)

    # 4. Compile with moderate learning rate for stable convergence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Create tf.data pipeline with fast horizontal flip augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=4096, seed=42)
    train_ds = train_ds.map(
        lambda img, lbl: (tf.image.random_flip_left_right(img), lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(64).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            improved_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 6. Train model
    print("\nStarting training with balanced class weights...")
    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=7,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    t_end = time.time()
    print(f"\nTraining completed in {t_end - t_start:.2f}s.")
    print(f"Best improved model checkpoint saved to: {improved_model_path}")

if __name__ == "__main__":
    main()
