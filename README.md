# Fake News Detection System

A beginner-friendly **OTML (Optimization Techniques and Machine Learning)** project that uses **TF-IDF text features** and **Logistic Regression** to classify a news statement as **REAL** or **FAKE**.

## Features
- Machine-learning based text classification
- TF-IDF feature extraction
- Logistic Regression classifier
- Flask web interface
- Confidence score
- Small included dataset for immediate testing
- Easy to extend with a larger dataset

## Project Structure

```text
fake_news_detection_system/
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── README.md
├── data/
│   └── news.csv
└── templates/
    └── index.html
```

## How to Run

### 1. Install Python
Use Python 3.10+.

### 2. Open terminal in this project folder

### 3. Create a virtual environment (recommended)

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Train the model
```bash
python train_model.py
```

This creates/updates `model.pkl`.

### 6. Start the website
```bash
python app.py
```

Open:
```text
http://127.0.0.1:5000
```

## Important Note
The included CSV is a **small demonstration dataset**, so the model is suitable for a college project demonstration, not for real-world fact checking. For a stronger project, replace it with a larger verified dataset and retrain the model.

## GitHub Upload

1. Create a new GitHub repository named `fake-news-detection-system`.
2. Upload all files and folders from this project.
3. Commit the changes.
4. Add the project description:
   `ML-based Fake News Detection System using TF-IDF, Logistic Regression and Flask.`
5. Run the project locally using the steps above.

## Suggested OTML Concepts to Explain

- Dataset and preprocessing
- TF-IDF as feature extraction
- Logistic Regression as classification
- Train/test split
- Accuracy and classification report
- Model optimization using hyperparameters
- Prediction confidence
