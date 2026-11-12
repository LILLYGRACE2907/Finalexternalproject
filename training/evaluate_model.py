import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def load_labels(labels_path):
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
    return DEFAULT_EMOTIONS.copy()


def evaluate_model(model_path=None, test_dir=None, output_path=None, batch_size=128):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = model_path or os.path.join(base_dir, 'models', 'emotion_model.h5')
    test_dir = test_dir or os.path.join(base_dir, 'dataset', 'test')
    labels_path = os.path.join(base_dir, 'models', 'emotion_labels.txt')
    output_path = output_path or os.path.join(base_dir, 'results', 'reports', 'evaluation_report.json')

    print("=" * 60)
    print("EMOTION RECOGNITION MODEL EVALUATION")
    print("=" * 60)
    print(f"Model file   : {model_path}")
    print(f"Test dataset : {test_dir}")
    print(f"Labels file  : {labels_path}")
    print(f"Batch size   : {batch_size}")
    print("-" * 60)

    if not os.path.isfile(model_path):
        print(f"[-] ERROR: Model file not found at: {model_path}")
        return None

    if not os.path.isdir(test_dir):
        print(f"[-] ERROR: Dataset directory not found at: {test_dir}")
        return None

    labels = load_labels(labels_path)
    print(f"[+] Loaded {len(labels)} emotion labels: {labels}")

    # Load model
    print("[*] Loading TensorFlow / Keras model...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[+] Model loaded successfully.")
        print(f"    Input shape : {model.input_shape}")
        print(f"    Output shape: {model.output_shape}")
    except Exception as exc:
        print(f"[-] ERROR: Failed to load model: {exc}")
        return None

    # Image generator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    total_samples = test_generator.samples
    print(f"[+] Found {total_samples} test images across {len(test_generator.class_indices)} subdirectories.")
    print(f"    Directory class indices: {test_generator.class_indices}")

    # Map generator classes to label indices
    # Sort labels to match alphabetical order of flow_from_directory
    sorted_dir_classes = sorted(test_generator.class_indices.keys())
    print(f"    Sorted directory classes: {sorted_dir_classes}")

    print("[*] Running inference on test dataset...")
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    num_classes = len(labels)
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    total_correct = int(np.trace(cm))
    total_evaluated = int(np.sum(cm))
    overall_accuracy = float(total_correct / total_evaluated) if total_evaluated > 0 else 0.0

    # Per-class metrics
    metrics_per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    print("\n" + "=" * 70)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = int(np.sum(cm[:, i])) - tp
        fn = int(np.sum(cm[i, :])) - tp
        support = int(np.sum(cm[i, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        metrics_per_class[label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': support
        }

        print(f"{label:<12} | {precision:>9.4f}  | {recall:>9.4f}  | {f1:>9.4f}  | {support:>8}")

    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes

    print("-" * 70)
    print(f"{'Accuracy':<12} | {'':<10} | {'':<10} | {overall_accuracy:>9.4f}  | {total_evaluated:>8}")
    print(f"{'Macro Avg':<12} | {macro_precision:>9.4f}  | {macro_recall:>9.4f}  | {macro_f1:>9.4f}  | {total_evaluated:>8}")
    print("=" * 70)

    # Print Confusion Matrix
    print("\nCONFUSION MATRIX (Rows: True, Columns: Predicted):")
    header = "       " + " ".join([f"{l[:5]:>7}" for l in labels])
    print(header)
    for i, label in enumerate(labels):
        row_str = f"{label[:6]:<6} " + " ".join([f"{cm[i, j]:>7}" for j in range(num_classes)])
        print(row_str)

    # Compile report dictionary
    report = {
        'model_name': os.path.basename(model_path),
        'model_path': model_path,
        'dataset_evaluated': test_dir,
        'total_samples': total_evaluated,
        'overall_accuracy': round(overall_accuracy, 4),
        'macro_avg': {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1_score': round(macro_f1, 4)
        },
        'metrics_per_class': metrics_per_class,
        'confusion_matrix': cm.tolist(),
        'labels': labels
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print(f"\n[+] Detailed evaluation report saved to: {output_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition CNN Model")
    parser.add_argument('--model', type=str, default=None, help="Path to .h5 model file")
    parser.add_argument('--dataset', type=str, default=None, help="Path to test dataset directory")
    parser.add_argument('--batch-size', type=int, default=128, help="Inference batch size")
    parser.add_argument('--output', type=str, default=None, help="Output JSON report file path")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        test_dir=args.dataset,
        output_path=args.output,
        batch_size=args.batch_size
    )
