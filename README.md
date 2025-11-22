# Antigravity Teachable Machine

A web-based multi-class image classifier that allows users to train Machine Learning models directly from their browser using a webcam.

## Features
- **Data Collection**: Capture images from your webcam for multiple custom classes.
- **Multi-Model Training**: Trains three models simultaneously:
  - Logistic Regression (Scikit-learn)
  - Random Forest (Scikit-learn)
  - CNN (TensorFlow/Keras)
- **Live Training Progress**: Real-time feedback on training status.
- **Evaluation Metrics**: View accuracy for each model.
- **Real-time Inference**: Test models live with your webcam.

## Architecture

### Backend (`app/`)
Built with **FastAPI**.
- **`main.py`**: API endpoints for uploading data, triggering training, and running inference. Manages global training state.
- **`trainers/`**: Contains logic for training different model types.
  - `logistic_regression.py`: Flattens images, trains LR.
  - `random_forest.py`: Flattens images, trains RF.
  - `cnn.py`: Uses 2D images, trains a Convolutional Neural Network.
- **`inference/`**: Handles loading saved models and generating predictions.
- **`data/`**: Utilities for image processing (resizing, normalization).

### Frontend (`app/static/`)
Built with **Vanilla HTML/CSS/JS**.
- **`index.html`**: Main layout.
- **`style.css`**: Premium dark-mode styling.
- **`script.js`**: Handles webcam streams, API communication, and UI updates.

## Installation

1. **Prerequisites**: Python 3.8+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Server**:
   ```bash
   uvicorn app.main:app --reload
   ```
2. **Open Web App**:
   Navigate to `http://127.0.0.1:8000` in your browser.

3. **Workflow**:
   - **Add Classes**: Create at least 2 classes (e.g., "Rock", "Paper").
   - **Capture Data**: Use "Hold to Record" to capture ~20-50 images per class.
   - **Train**: Click "Train Model" and wait for completion.
   - **Predict**: Use the "Preview" panel to see live predictions from all 3 models.

## File Structure
```
d:/Task 3/
  app/
    data/          # Stored images
    trainers/      # Training logic
    models/        # Saved models (.pkl, .h5)
    inference/     # Prediction logic
    static/        # Frontend assets
    main.py        # FastAPI app
  requirements.txt
```
