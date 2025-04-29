import tensorflow as tf
import os

def convert_h5_to_saved_model():
    try:
        # Load the H5 model
        h5_path = '../toxic_comment_model.h5'
        saved_model_path = '../saved_model'
        
        # Load model
        model = tf.keras.models.load_model(h5_path, compile=False)
        
        # Save in SavedModel format
        tf.saved_model.save(model, saved_model_path)
        print("Model converted and saved successfully!")
        return True
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False

if __name__ == '__main__':
    convert_h5_to_saved_model()