import os
import pandas as pd
import whisper
import torch
import librosa
import numpy as np
import pickle
import gc
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory management function
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Check if GPU is available and has sufficient memory
def get_device():
    if torch.cuda.is_available():
        try:
            # Try to allocate a small tensor to check memory
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            return 'cuda'
        except RuntimeError:
            logger.warning("CUDA memory error, falling back to CPU")
            clear_gpu_memory()
            return 'cpu'
    return 'cpu'

# Initialize device
device = get_device()
logger.info(f"Using device: {device}")

# Load models with error handling
try:
    # Load Whisper model for transcription
    whisper_model = whisper.load_model("base", device=device)
    
    # Load grammar scoring model
    with open("outputs/grammar_scorer.pkl", "rb") as f:
        grammar_model = pickle.load(f)
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper"""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        clear_gpu_memory()  # Clear memory before transcription
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("CUDA out of memory, falling back to CPU")
            clear_gpu_memory()
            # Move model to CPU temporarily
            whisper_model.to('cpu')
            result = whisper_model.transcribe(audio_path)
            # Move back to original device
            whisper_model.to(device)
            return result["text"]
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return f"Error: {str(e)}"

def extract_audio_features(audio_path):
    """Extract audio features using librosa"""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        y, sr = librosa.load(audio_path, sr=None)
        return {
            "duration": librosa.get_duration(y=y, sr=sr),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "rmse": np.mean(librosa.feature.rms(y=y))
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        # Return default values if extraction fails
        return {
            "duration": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid": 0.0,
            "rmse": 0.0
        }

def extract_text_features(text):
    """Extract text features for grammar scoring"""
    try:
        words = text.split()
        return {
            "num_words": len(words),
            "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
            "text_length": len(text)
        }
    except Exception as e:
        logger.error(f"Error extracting text features: {str(e)}")
        # Return default values if extraction fails
        return {
            "num_words": 0,
            "avg_word_len": 0,
            "text_length": 0
        }

def process_audio_file(audio_path):
    """Process a single audio file and return results"""
    try:
        # Get filename from path
        audio_filename = os.path.basename(audio_path)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        
        # Extract features
        audio_feats = extract_audio_features(audio_path)
        text_feats = extract_text_features(transcript)
        features = {**audio_feats, **text_feats}
        
        # Score grammar
        df = pd.DataFrame([features])
        grammar_score = grammar_model.predict(df)[0]
        
        return {
            "filename": audio_filename,
            "transcript": transcript,
            "label": grammar_score
        }
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        return {
            "filename": os.path.basename(audio_path),
            "transcript": f"Error: {str(e)}",
            "label": 0
        }

def process_directory(input_dir, output_file):
    """Process all audio files in a directory"""
    try:
        # Get all audio files in the directory
        audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Create results list
        results = []
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            audio_path = os.path.join(input_dir, audio_file)
            result = process_audio_file(audio_path)
            results.append(result)
            
            # Clear memory after each file
            clear_gpu_memory()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise

def main():
    """Main function to process audio files"""
    parser = argparse.ArgumentParser(description="Process audio files and output grammar scores")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="grammar_scores.csv", help="Output CSV file")
    parser.add_argument("--directory", "-d", action="store_true", help="Process all audio files in the input directory")
    
    args = parser.parse_args()
    
    try:
        if args.directory:
            # Process all audio files in the directory
            output_file = process_directory(args.input, args.output)
        else:
            # Process a single audio file
            result = process_audio_file(args.input)
            
            # Create results DataFrame
            results_df = pd.DataFrame([result])
            
            # Save results to CSV
            results_df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
            
            output_file = args.output
        
        print(f"Processing complete. Results saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 