import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the data
def load_data(file_path):
    """Load and preprocess the training data."""
    df = pd.read_csv(file_path)
    return df

# Initialize and fit the TF-IDF vectorizer
def create_tfidf_features(texts, max_features=50000):
    """Create TF-IDF features from text data."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Train models for each toxicity type
def train_model(X_train, y_train, X_test, y_test, label):
    """Train an XGBoost model for a specific toxicity label."""
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for {label}:")
    print(classification_report(y_test, y_pred))
    
    return model

def save_models(models, vectorizer):
    """Save trained models and vectorizer."""
    os.makedirs('models', exist_ok=True)
    
    # Save vectorizer
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    # Save models
    for label, model in models.items():
        joblib.dump(model, f'models/model_{label}.joblib')

def main():
    # Define toxicity labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Load data
    print("Loading data...")
    df = load_data('data/train.csv')  # Update path as needed
    
    # Create features
    print("Creating TF-IDF features...")
    X, vectorizer = create_tfidf_features(df['comment_text'])
    
    # Train models for each toxicity type
    models = {}
    for label in labels:
        print(f"\nTraining model for {label}...")
        y = df[label]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate model
        model = train_model(X_train, y_train, X_test, y_test, label)
        models[label] = model
    
    # Save models and vectorizer
    print("\nSaving models...")
    save_models(models, vectorizer)
    print("Training complete! Models saved in 'models' directory.")

if __name__ == '__main__':
    main() 