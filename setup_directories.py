import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "dataset",
        "dataset/audios_test",
        "outputs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")
    
    # Check if test.csv exists
    if not os.path.exists("dataset/test.csv"):
        logger.warning("test.csv not found in dataset directory. Please ensure it exists before running the batch processor.")
    else:
        logger.info("test.csv found in dataset directory.")
    
    # Check if grammar_scorer.pkl exists
    if not os.path.exists("outputs/grammar_scorer.pkl"):
        logger.warning("grammar_scorer.pkl not found in outputs directory. Please ensure it exists before running the batch processor.")
    else:
        logger.info("grammar_scorer.pkl found in outputs directory.")

if __name__ == "__main__":
    create_directories()
    print("Directory setup complete. Please ensure all required files are in place before running batch_grammar_assistant.py") 