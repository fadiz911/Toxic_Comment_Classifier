import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

# Initialize the models and components
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.7)
xgb = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1))

# Label columns for multi-label classification
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def save_model(model, vectorizer, output_dir='models/'):
    """
    Save the trained model and vectorizer
    
    Args:
        model: Trained model object
        vectorizer: Fitted TfidfVectorizer object
        output_dir: Directory to save the model files
    """
    # Save the model
    joblib.dump(model, f'{output_dir}toxic_comment_model.joblib')
    
    # Save the vectorizer
    joblib.dump(vectorizer, f'{output_dir}vectorizer.joblib')
    
    print(f"Model and vectorizer saved to {output_dir}")

def load_model(model_dir='models/'):
    """
    Load the saved model and vectorizer
    
    Args:
        model_dir: Directory containing the saved model files
        
    Returns:
        model: Loaded model object
        vectorizer: Loaded vectorizer object
    """
    # Load the model
    model = joblib.load(f'{model_dir}toxic_comment_model.joblib')
    
    # Load the vectorizer
    vectorizer = joblib.load(f'{model_dir}vectorizer.joblib')
    
    return model, vectorizer

def predict(text, model, vectorizer):
    """
    Make predictions on new text
    
    Args:
        text: Text to classify
        model: Loaded model object
        vectorizer: Loaded vectorizer object
        
    Returns:
        predictions: Dictionary containing prediction probabilities for each class
    """
    # Transform the text using the vectorizer
    X = vectorizer.transform([text])
    
    # Get predictions
    pred_proba = model.predict_proba(X)
    
    # Create dictionary of predictions
    predictions = {}
    for i, label in enumerate(label_cols):
        predictions[label] = pred_proba[i][0][1]
        
    return predictions

if __name__ == "__main__":
    # Example usage
    text = "This is a test comment"
    
    # Load the model and vectorizer
    model, vectorizer = load_model()
    
    # Make predictions
    predictions = predict(text, model, vectorizer)
    print("\nPredictions:")
    for label, prob in predictions.items():
        print(f"{label}: {prob:.4f}") 