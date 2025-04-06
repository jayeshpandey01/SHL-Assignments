import gradio as gr
import whisper
import torch
import librosa
import numpy as np
import pandas as pd
import pyttsx3
import pickle
import os
import tempfile
import gc
import logging
from transformers import pipeline, BitsAndBytesConfig

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
        device=0 if device == 'cuda' else -1,
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
        raise

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
        raise

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
        raise

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

def text_to_speech(text, output_path="corrected_output.wav"):
    """Convert text to speech using pyttsx3"""
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return output_path
    except Exception as e:
        logger.error(f"Error converting text to speech: {str(e)}")
        raise

def process_input(audio_file, text_input):
    """Main function to process audio or text input"""
    try:
        # Handle empty inputs
        if audio_file is None and not text_input:
            return "Please provide either an audio file or text input.", None, None, None, None
        
        # Process audio input if provided
        transcript = ""
        if audio_file is not None:
            try:
                transcript = transcribe_audio(audio_file)
            except Exception as e:
                logger.error(f"Error in audio transcription: {e}")
                return f"Error processing audio: {str(e)}", None, None, None, None
        
        # Use text input if provided, otherwise use transcript
        text_to_process = text_input if text_input else transcript
        
        if not text_to_process:
            return "No text to process.", None, None, None, None
        
        # Correct grammar
        corrected_text = correct_grammar(text_to_process)
        
        # Extract features for grammar scoring
        if audio_file:
            try:
                # Combine audio and text features
                audio_feats = extract_audio_features(audio_file)
                text_feats = extract_text_features(text_to_process)
                features = {**audio_feats, **text_feats}
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                return f"Error extracting features: {str(e)}", transcript, text_to_process, None, None
        else:
            # Use dummy audio features if no audio
            text_feats = extract_text_features(text_to_process)
            dummy_audio_feats = {
                "duration": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
                "rmse": 0.0
            }
            features = {**dummy_audio_feats, **text_feats}
        
        # Score grammar
        df = pd.DataFrame([features])
        grammar_score = grammar_model.predict(df)[0]
        
        # Generate audio output
        audio_output = text_to_speech(corrected_text)
        
        return transcript, text_to_process, corrected_text, grammar_score, audio_output
    
    except Exception as e:
        logger.error(f"Error in process_input: {e}")
        return f"An error occurred: {str(e)}", None, None, None, None
    finally:
        clear_gpu_memory()  # Clean up memory after processing

# Create Gradio interface
demo = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Record or Upload Audio"),
        gr.Textbox(label="Or Enter Text Directly", placeholder="Type your text here...")
    ],
    outputs=[
        gr.Textbox(label="Transcription (if audio provided)"),
        gr.Textbox(label="Original Text"),
        gr.Textbox(label="Grammar-Corrected Text"),
        gr.Number(label="Grammar Score (0-5)"),
        gr.Audio(label="Corrected Audio Output")
    ],
    title="Grammar Assistant",
    description="Upload an audio file or enter text to get grammar correction and scoring. The app transcribes audio, corrects grammar, scores the grammar quality, and provides both text and voice output.",
    examples=[
        ["People in the market are selling just about anything and everything. You can hear everyone screaming and talking over each other, making offers."],
        ["The crowded market scene makes me want to run out of the door as soon as possible, and I picture this happening midday."]
    ]
)

if __name__ == "__main__":
    demo.launch() 