from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__, static_folder='..', static_url_path='')

# Constants
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Initialize models and vectorizer as None
models = {}
vectorizer = None

def load_models_and_vectorizer():
    """Load the TF-IDF vectorizer and all toxicity models"""
    global models, vectorizer
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, '..', 'models')
        
        # Load vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            print("Vectorizer loaded successfully")
        else:
            print("No saved vectorizer found")
            return False
        
        # Load all models
        for label in LABELS:
            model_path = os.path.join(models_dir, f'model_{label}.joblib')
            if os.path.exists(model_path):
                models[label] = joblib.load(model_path)
                print(f"Model for {label} loaded successfully")
            else:
                print(f"No saved model found for {label}")
                return False
        
        return True

    except Exception as e:
        print(f"Error loading models or vectorizer: {str(e)}")
        return False

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global models, vectorizer
    
    try:
        # Load models and vectorizer if not loaded
        if not models or vectorizer is None:
            success = load_models_and_vectorizer()
            if not success:
                return jsonify({'success': False, 'error': 'Failed to load models or vectorizer'})
        
        # Get the text from the request
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})

        # Transform the text using TF-IDF vectorizer
        X = vectorizer.transform([text])
        
        # Get predictions from all models
        predictions = {}
        for label in LABELS:
            # Get probability of positive class (toxic)
            prob = models[label].predict_proba(X)[0][1]
            predictions[label] = float(prob)  # Convert numpy float to Python float for JSON
        
        # Create response
        result = {
            'success': True,
            'predictions': predictions
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Try to load the models and vectorizer at startup
    load_models_and_vectorizer()
    app.run(debug=True, port=5000)