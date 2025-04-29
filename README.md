# Toxic Comment Classifier
![Toxic Comment Classifier Logo](data/logo.png)

This project implements a machine learning system for classifying toxic comments across multiple categories of toxicity. It uses XGBoost and TF-IDF vectorization to create models that can identify different types of toxic content in text.

## Key Features

- Multi-label toxicity classification
- TF-IDF vectorization for text feature extraction
- XGBoost classifier implementation
- Model performance evaluation with classification reports
- Automated model saving and loading functionality
- Web interface for real-time toxicity classification

## System Requirements

- Python 3.7 or higher
- Required Python packages (specified in requirements.txt)
- Sufficient RAM for processing text data and training models
- Disk space for storing models and vectorizers

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd toxic-comment-classifier
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Format

The training data should be provided in a CSV file (`data/train.csv`) with the following columns:
- `comment_text`: The text content to be classified
- `toxic`: Binary label (0 or 1)
- `severe_toxic`: Binary label (0 or 1)
- `obscene`: Binary label (0 or 1)
- `threat`: Binary label (0 or 1)
- `insult`: Binary label (0 or 1)
- `identity_hate`: Binary label (0 or 1)

## Training the Model

To train the models, run:
```bash
python train_model.py
```

This will:
1. Load the training data from `data/train.csv`
2. Create TF-IDF features from the comment text
3. Train separate XGBoost models for each toxicity category
4. Save the trained models and vectorizer in the `models` directory

## Running the Web Application

After training the models, you can start the web application by running:
```bash
python src/app.py
```

The application will:
1. Load the trained models and vectorizer
2. Start a Flask web server on port 5000
3. Serve a web interface at `http://localhost:5000`
4. Provide an API endpoint at `http://localhost:5000/predict` for real-time toxicity classification

To use the web interface:
1. Open your web browser and navigate to `http://localhost:5000`
2. Enter the text you want to analyze in the provided text box
3. Click the "Classify" button to get toxicity predictions
4. The results will show the probability of each toxicity category

## Model Files

After training, the following files will be created in the `models` directory:
- `tfidf_vectorizer.joblib`: The fitted TF-IDF vectorizer
- `model_toxic.joblib`: Model for toxic classification
- `model_severe_toxic.joblib`: Model for severe toxic classification
- `model_obscene.joblib`: Model for obscene content classification
- `model_threat.joblib`: Model for threat classification
- `model_insult.joblib`: Model for insult classification
- `model_identity_hate.joblib`: Model for identity hate classification

## Performance Metrics

The training script will output classification reports for each toxicity category, including:
- Precision
- Recall
- F1-score
- Support
