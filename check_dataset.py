
import os
import hashlib

classes = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

def image_hashes(folder):
    hashes = set()

    if not os.path.isdir(folder):
        return hashes

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        with open(path, "rb") as file:
            hashes.add(hashlib.sha256(file.read()).hexdigest())

    return hashes

for emotion in classes:
    train = image_hashes(os.path.join("dataset", "train", emotion))
    test = image_hashes(os.path.join("dataset", "test", emotion))

    common = train & test

    print(
        f"{emotion}: train={len(train)}, "
        f"test={len(test)}, same_images={len(common)}"
    )
