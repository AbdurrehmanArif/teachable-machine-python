# Teachable Machine - Python

A web-based machine learning application inspired by Google's Teachable Machine. Train image classification models directly in your browser using webcam or file uploads.

## Features

- 📸 **Webcam Integration**: Capture training images in real-time
- 📁 **File/Folder Upload**: Upload images or entire folders
- 🤖 **Multiple Models**: Train Logistic Regression, Random Forest, and CNN models
- 📊 **Evaluation Metrics**: View accuracy and confusion matrices
- 🔮 **Live Prediction**: Real-time predictions using your webcam
- 🎨 **Modern UI**: Clean, responsive interface

## Tech Stack

- **Backend**: FastAPI
- **ML Models**: Scikit-learn, TensorFlow/Keras
- **Frontend**: Vanilla JavaScript, HTML, CSS

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbdurrehmanArif/teachable-machine-python.git
   cd teachable-machine-python
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:8000
   ```

## Deployment

### Option 1: Render (Recommended - Free)

1. Go to [Render.com](https://render.com) and sign up with GitHub
2. Click **"New +"** → **"Web Service"**
3. Select this repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Click **"Create Web Service"**

### Option 2: Railway

1. Go to [Railway.app](https://railway.app)
2. Click **"Deploy from GitHub repo"**
3. Select this repository
4. Railway will auto-detect and deploy

### Option 3: Heroku

1. Install Heroku CLI
2. Run:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Usage

1. **Create Classes**: Add class names (e.g., "Cat", "Dog")
2. **Collect Data**: 
   - Hold "Record" button to capture webcam images
   - Or upload files/folders
3. **Train Model**: Click "Train Model" button
4. **View Results**: See accuracy and confusion matrix
5. **Live Prediction**: Toggle "Live Prediction" to test your model

## Project Structure

```
teachable-machine-python/
├── app/
│   ├── data/           # Data utilities
│   ├── trainers/       # ML model trainers
│   ├── inference/      # Prediction logic
│   ├── models/         # Saved models
│   ├── static/         # Frontend files
│   └── main.py         # FastAPI server
├── requirements.txt
├── Procfile           # For deployment
└── run.bat            # Windows run script
```

## Requirements

- Python 3.8+
- Webcam (for live capture)
- Modern browser with webcam permissions

## Notes

- **Single Class Training**: Supported (uses DummyClassifier)
- **Minimum Images**: At least 5 images per class recommended
- **Model Storage**: Models saved in `app/models/`
- **Data Storage**: Images saved in `app/data/images/`

## License

MIT License

## Author

**Abdurrehman Arif**
- GitHub: [@AbdurrehmanArif](https://github.com/AbdurrehmanArif)

---

⚠️ **Important**: This is a FastAPI application, NOT a Streamlit app. Do not deploy on Streamlit Cloud.
