import os
import cv2
import numpy as np

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def prepare_dataset_directories():
    """
    Creates dataset directory structure for FER training & testing:
    dataset/
      ├── train/
      │   ├── angry/ ...
      └── test/
          ├── angry/ ...
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    for category in EMOTION_CLASSES:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
    print(f"[+] Dataset directory structure prepared at: {dataset_dir}")
    return train_dir, test_dir

def generate_sample_synthetic_data(samples_per_class=10):
    """
    Generate synthetic 48x48 sample grayscale images for each class
    if real dataset (FER2013) is not yet populated.
    """
    train_dir, test_dir = prepare_dataset_directories()
    
    for category in EMOTION_CLASSES:
        cat_train = os.path.join(train_dir, category)
        cat_test = os.path.join(test_dir, category)
        
        # Check if directory already has images
        if len(os.listdir(cat_train)) == 0:
            for i in range(samples_per_class):
                # Generate sample 48x48 image with random patterns
                img = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
                cv2.imwrite(os.path.join(cat_train, f"sample_{i}.jpg"), img)
                
        if len(os.listdir(cat_test)) == 0:
            for i in range(max(2, samples_per_class // 5)):
                img = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
                cv2.imwrite(os.path.join(cat_test, f"sample_{i}.jpg"), img)
                
    print("[+] Synthetic sample dataset generated successfully.")

if __name__ == '__main__':
    generate_sample_synthetic_data()
