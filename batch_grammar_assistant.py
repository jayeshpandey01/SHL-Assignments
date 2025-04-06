import os
import pandas as pd
import whisper
import torch
import librosa
import numpy as np
import pickle
import gc
import logging
from transformers import pipeline, BitsAndBytesConfig
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
    # Load Whisper model for transcription with memory optimization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    whisper_model = whisper.load_model("base", device=device)
    
    # Load grammar scoring model
    with open("outputs/grammar_scorer.pkl", "rb") as f:
        grammar_model = pickle.load(f)
    
    # Load grammar correction model with memory optimization
    corrector = pipeline(
        "text2text-generation",
        model="vennify/t5-base-grammar-correction",
        model_kwargs={"quantization_config": bnb_config}
    )
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
            "grammar_score_model": 0,  # Required by the model
            "num_words": len(words),
            "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
            "text_length": len(text)
        }
    except Exception as e:
        logger.error(f"Error extracting text features: {str(e)}")
        # Return default values if extraction fails
        return {
            "grammar_score_model": 0,
            "num_words": 0,
            "avg_word_len": 0,
            "text_length": 0
        }

def correct_grammar(text):
    """Correct grammar using the Hugging Face model"""
    try:
        # Split long text into smaller chunks if needed
        max_chunk_length = 500
        chunks = []
        
        # Simple chunking by sentences
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If text is short enough, process it as a single chunk
        if len(chunks) == 1:
            clear_gpu_memory()  # Clear memory before correction
            result = corrector(text, max_length=512, do_sample=False)[0]['generated_text']
            return result
        else:
            # Process each chunk separately and combine results
            corrected_chunks = []
            for chunk in chunks:
                clear_gpu_memory()  # Clear memory before each correction
                corrected_chunk = corrector(chunk, max_length=512, do_sample=False)[0]['generated_text']
                corrected_chunks.append(corrected_chunk)
            
            # Combine chunks
            final_result = " ".join(corrected_chunks)
            return final_result
    except Exception as e:
        logger.error(f"Error correcting grammar: {str(e)}")
        # Return original text if correction fails
        return text

def process_audio_file(audio_filename, audio_dir="dataset/audios_test"):
    """Process a single audio file and return results"""
    try:
        # Construct full path to audio file
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        
        # Extract features
        audio_feats = extract_audio_features(audio_path)
        text_feats = extract_text_features(transcript)
        features = {**audio_feats, **text_feats}
        
        # Score grammar
        df = pd.DataFrame([features])
        grammar_score = grammar_model.predict(df)[0]
        
        # Correct grammar
        corrected_text = correct_grammar(transcript)
        
        return {
            "filename": audio_filename,
            "transcript": transcript,
            "grammar_score": grammar_score,
            "corrected_text": corrected_text
        }
    except Exception as e:
        logger.error(f"Error processing {audio_filename}: {str(e)}")
        return {
            "filename": audio_filename,
            "transcript": f"Error: {str(e)}",
            "grammar_score": 0,
            "corrected_text": f"Error: {str(e)}"
        }

def main():
    """Main function to process all audio files in test.csv"""
    try:
        # Read test.csv file
        test_df = pd.read_csv("dataset/test.csv")
        logger.info(f"Found {len(test_df)} audio files to process")
        
        # Create results list
        results = []
        
        # Process each audio file
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing audio files"):
            audio_filename = row["filename"]
            result = process_audio_file(audio_filename)
            results.append(result)
            
            # Clear memory after each file
            clear_gpu_memory()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        output_path = "dataset/grammar_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    output_file = main()
    print(f"Processing complete. Results saved to: {output_file}") 