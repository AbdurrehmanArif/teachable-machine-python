import joblib
import tensorflow as tf
import numpy as np
import json
import os
from app.data.utils import load_and_preprocess_image

class Predictor:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.models = {}
        self.classes = []
        self.load_models()
        
    def load_models(self):
        # Load classes
        classes_path = os.path.join(self.models_dir, "classes.json")
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                self.classes = json.load(f)
        
        # Load LR
        lr_path = os.path.join(self.models_dir, "logistic_regression.pkl")
        if os.path.exists(lr_path):
            self.models["logistic_regression"] = joblib.load(lr_path)
            
        # Load RF
        rf_path = os.path.join(self.models_dir, "random_forest.pkl")
        if os.path.exists(rf_path):
            self.models["random_forest"] = joblib.load(rf_path)
            
        # Load CNN
        cnn_path = os.path.join(self.models_dir, "cnn.h5")
        if os.path.exists(cnn_path):
            self.models["cnn"] = tf.keras.models.load_model(cnn_path)
            
    def predict(self, image_path):
        results = {}
        
        # Preprocess for LR/RF (Flattened)
        img_flat = load_and_preprocess_image(image_path, flatten=True)
        if img_flat is not None:
            img_flat = img_flat.reshape(1, -1)
            
            if "logistic_regression" in self.models:
                try:
                    pred_idx = self.models["logistic_regression"].predict(img_flat)[0]
                    # Handle case where predict returns class label directly or index
                    if isinstance(pred_idx, (np.integer, int)) and pred_idx < len(self.classes):
                         results["logistic_regression"] = self.classes[pred_idx]
                    else:
                         results["logistic_regression"] = str(pred_idx)
                except Exception as e:
                    print(f"LR Prediction Error: {e}")
                    results["logistic_regression"] = "Error"
                
            if "random_forest" in self.models:
                try:
                    pred_idx = self.models["random_forest"].predict(img_flat)[0]
                    if isinstance(pred_idx, (np.integer, int)) and pred_idx < len(self.classes):
                         results["random_forest"] = self.classes[pred_idx]
                    else:
                         results["random_forest"] = str(pred_idx)
                except Exception as e:
                    print(f"RF Prediction Error: {e}")
                    results["random_forest"] = "Error"

        # Preprocess for CNN (Not flattened)
        img_cnn = load_and_preprocess_image(image_path, flatten=False)
        if img_cnn is not None:
            img_cnn = np.expand_dims(img_cnn, axis=0) # Add batch dimension
            
            if "cnn" in self.models:
                try:
                    pred_prob = self.models["cnn"].predict(img_cnn)
                    # If only 1 class, pred_prob might be weird, but argmax usually works
                    if pred_prob.shape[1] > 1:
                        pred_idx = np.argmax(pred_prob, axis=1)[0]
                    else:
                        # If output is 1D or single value, assume class 0
                        pred_idx = 0
                        
                    if pred_idx < len(self.classes):
                        results["cnn"] = self.classes[pred_idx]
                    else:
                        results["cnn"] = str(pred_idx)
                except Exception as e:
                    print(f"CNN Prediction Error: {e}")
                    results["cnn"] = "Error"
                
        return results
