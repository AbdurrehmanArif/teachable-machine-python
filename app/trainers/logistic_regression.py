from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import numpy as np

class LogisticRegressionTrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        
    def train(self, X, y, save_path):
        # Check if only 1 class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            from sklearn.dummy import DummyClassifier
            # Use DummyClassifier for single class case
            self.model = DummyClassifier(strategy="constant", constant=unique_classes[0])
            self.model.fit(X, y)
            
            joblib.dump(self.model, save_path)
            return {"accuracy": 1.0, "confusion_matrix": [[len(y)]]}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Save
        joblib.dump(self.model, save_path)
        
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm
        }
