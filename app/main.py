from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import json
import uuid
from app.data.utils import load_dataset
from app.trainers.logistic_regression import LogisticRegressionTrainer
from app.trainers.random_forest import RandomForestTrainer
from app.trainers.cnn import CNNTrainer
from app.inference.predictor import Predictor

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = "app/data/images"
MODELS_DIR = "app/models"
TEMP_DIR = "app/data/temp"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Global State
training_state = {
    "status": "idle", # idle, training, completed, error
    "progress": 0,
    "message": "Ready to train",
    "results": {}
}

predictor = Predictor(MODELS_DIR)

@app.get("/state")
def get_state():
    return training_state

@app.get("/classes")
def get_classes():
    classes = []
    if os.path.exists(DATA_DIR):
        for d in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, d)
            if os.path.isdir(path):
                images = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
                count = len(images)
                # Return top 12 images for preview
                classes.append({"name": d, "count": count, "images": images[:12]})
    return classes

@app.post("/create_class/{class_name}")
def create_class(class_name: str):
    path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return {"message": f"Class {class_name} created"}

@app.delete("/delete_class/{class_name}")
def delete_class(class_name: str):
    path = os.path.join(DATA_DIR, class_name)
    if os.path.exists(path):
        shutil.rmtree(path)
        return {"message": f"Class {class_name} deleted"}
    raise HTTPException(status_code=404, detail="Class not found")

@app.post("/upload/{class_name}")
async def upload_image(class_name: str, file: UploadFile = File(...)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(class_path, filename)
    
    # Validate image
    try:
        from PIL import Image
        img = Image.open(file.file)
        img.verify()
        file.file.seek(0) # Reset pointer
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": "Image uploaded", "filename": filename}

def train_models_task():
    global training_state, predictor
    training_state["status"] = "training"
    training_state["progress"] = 10
    training_state["message"] = "Loading dataset..."
    
    try:
        # Load Data
        X, y, classes = load_dataset(DATA_DIR, flatten=True) # Load flattened for LR/RF
        X_cnn, y_cnn, _ = load_dataset(DATA_DIR, flatten=False) # Load unflattened for CNN
        
        if len(classes) < 1:
            raise Exception("Need at least 1 class to train.")
            
        # Check minimum images per class
        for class_name in classes:
            class_dir = os.path.join(DATA_DIR, class_name)
            if len(os.listdir(class_dir)) < 5:
                raise Exception(f"Class '{class_name}' needs at least 5 images.")
            
        # Save classes
        with open(os.path.join(MODELS_DIR, "classes.json"), "w") as f:
            json.dump(classes, f)
            
        training_state["progress"] = 30
        training_state["message"] = "Training Logistic Regression..."
        
        # Train LR
        lr = LogisticRegressionTrainer()
        lr_res = lr.train(X, y, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
        training_state["results"]["logistic_regression"] = lr_res
        
        training_state["progress"] = 50
        training_state["message"] = "Training Random Forest..."
        
        # Train RF
        rf = RandomForestTrainer()
        rf_res = rf.train(X, y, os.path.join(MODELS_DIR, "random_forest.pkl"))
        training_state["results"]["random_forest"] = rf_res
        
        training_state["progress"] = 70
        training_state["message"] = "Training CNN..."
        
        # Train CNN
        cnn = CNNTrainer(num_classes=len(classes))
        cnn_res = cnn.train(X_cnn, y_cnn, os.path.join(MODELS_DIR, "cnn.h5"))
        training_state["results"]["cnn"] = cnn_res
        
        training_state["progress"] = 100
        training_state["status"] = "completed"
        training_state["message"] = "Training completed successfully!"
        
        # Reload predictor
        predictor.load_models()
        
    except Exception as e:
        training_state["status"] = "error"
        training_state["message"] = str(e)
        print(f"Training error: {e}")

@app.post("/train")
async def train(background_tasks: BackgroundTasks):
    if training_state["status"] == "training":
        return {"message": "Training already in progress", "error": True}
    
    # Reset state before starting
    training_state["status"] = "idle"
    training_state["progress"] = 0
    training_state["message"] = "Ready to train"
    training_state["results"] = {}
    
    background_tasks.add_task(train_models_task)
    return {"message": "Training started"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        predictions = predictor.predict(file_path)
        # Clean up
        os.remove(file_path)
        return predictions
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Serve Data Files
app.mount("/data", StaticFiles(directory="app/data/images"), name="data")

# Serve Static Files
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
