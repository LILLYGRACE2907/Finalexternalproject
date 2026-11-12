
import random
import hashlib
import shutil
from pathlib import Path

SOURCE = Path("dataset/train")
OUTPUT = Path("dataset_split")

CLASSES = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)

if not SOURCE.exists():
    raise SystemExit(f"Source folder not found: {SOURCE}")

if OUTPUT.exists():
    raise SystemExit(
        "dataset_split already exists. Rename or remove it "
        "before running this script again."
    )

for split in ["train", "val", "test"]:
    for emotion in CLASSES:
        (OUTPUT / split / emotion).mkdir(parents=True, exist_ok=True)

try:
    for emotion in CLASSES:
        folder = SOURCE / emotion

        if not folder.exists():
            raise SystemExit(f"Missing class folder: {folder}")

        unique_images = {}

        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            unique_images.setdefault(digest, path)

        images = list(unique_images.values())
        random.shuffle(images)

        total = len(images)
        if total < 10:
            raise SystemExit(
                f"Too few unique images for {emotion}: {total}"
            )

        train_end = int(total * 0.8)
        val_end = int(total * 0.9)

        groups = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, files in groups.items():
            for index, source_file in enumerate(files):
                destination = (
                    OUTPUT / split / emotion /
                    f"{index:06d}{source_file.suffix.lower()}"
                )
                shutil.copy2(source_file, destination)

        print(
            f"{emotion}: unique={total}, "
            f"train={len(groups['train'])}, "
            f"val={len(groups['val'])}, "
            f"test={len(groups['test'])}"
        )

except Exception:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    raise

print("\nDataset split completed successfully.")
print("Original dataset was not changed.")
print("Output folder:", OUTPUT)
