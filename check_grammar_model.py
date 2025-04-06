import pickle
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_grammar_model():
    """Check the grammar model and understand what columns it expects"""
    try:
        # Load grammar scoring model
        with open("outputs/grammar_scorer.pkl", "rb") as f:
            grammar_model = pickle.load(f)
        
        # Print model information
        logger.info(f"Model type: {type(grammar_model)}")
        
        # Try different feature combinations
        feature_sets = [
            # Set 1: Basic text features
            {
                "num_words": [10],
                "avg_word_len": [5.0],
                "text_length": [50]
            },
            # Set 2: Text features + audio features
            {
                "num_words": [10],
                "avg_word_len": [5.0],
                "text_length": [50],
                "duration": [1.0],
                "zero_crossing_rate": [0.1],
                "spectral_centroid": [1000.0],
                "rmse": [0.5]
            }
        ]
        
        # Try each feature set
        for i, features in enumerate(feature_sets):
            try:
                df = pd.DataFrame(features)
                logger.info(f"Trying feature set {i+1}: {list(features.keys())}")
                result = grammar_model.predict(df)
                logger.info(f"Prediction successful: {result}")
            except Exception as e:
                logger.error(f"Error with feature set {i+1}: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking grammar model: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_grammar_model()
    if success:
        print("Grammar model check completed successfully")
    else:
        print("Grammar model check failed") 