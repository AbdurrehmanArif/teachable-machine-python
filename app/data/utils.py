import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(image_path, target_size=(64, 64), flatten=False):
    """
    Loads an image, resizes it, and converts it to a numpy array.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize
        
        if flatten:
            img_array = img_array.flatten()
            
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_dataset(data_dir, target_size=(64, 64), flatten=False):
    """
    Loads the dataset from the data directory.
    Structure: data_dir/class_name/image.jpg
    """
    X = []
    y = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_indices = {c: i for i, c in enumerate(classes)}
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_data = load_and_preprocess_image(img_path, target_size, flatten)
            if img_data is not None:
                X.append(img_data)
                y.append(class_indices[class_name])
                
    return np.array(X), np.array(y), classes
