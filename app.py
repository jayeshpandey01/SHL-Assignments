from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import json
import base64
from werkzeug.utils import secure_filename
import whisper
import torch
import librosa
import numpy as np
import pandas as pd
import pyttsx3
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import language_tool_python

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Whisper model for transcription
whisper_model = whisper.load_model("base").to(device)

# Load grammar scoring model
with open("outputs/grammar_scorer.pkl", "rb") as f:
    grammar_model = pickle.load(f)

# Load advanced grammar correction models
print("Loading advanced grammar correction models...")

# Load T5 model for grammar correction
t5_corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

# Load BART model for grammar correction
bart_corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Load LanguageTool for additional grammar checking
language_tool = language_tool_python.LanguageTool('en-US')

# Load advanced model with memory optimizations
try:
    # Try to load the more advanced model with memory optimizations
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    advanced_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    advanced_model = AutoModelForSeq2SeqLM.from_pretrained(
        "prithivida/grammar_error_correcter_v1",
        device_map="auto",
        quantization_config=bnb_config
    )
    advanced_corrector = pipeline("text2text-generation", model=advanced_model, tokenizer=advanced_tokenizer)
    print("Advanced grammar correction model loaded successfully")
except Exception as e:
    print(f"Could not load advanced model: {str(e)}")
    print("Falling back to basic models")
    advanced_corrector = None

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper"""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
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
        print(f"Error extracting audio features: {str(e)}")
        raise

def extract_text_features(text):
    """Extract text features for grammar scoring"""
    words = text.split()
    
    # Use LanguageTool to check for grammar errors
    matches = language_tool.check(text)
    # We'll keep the error count for display purposes but won't include it in the model features
    
    return {
        "grammar_score_model": 0,  # Required by the model
        "num_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
        "text_length": len(text)
        # Removed 'num_errors' to match training data features
    }

def correct_grammar(text):
    """Correct grammar using multiple models and combine results"""
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
            # Use T5 model for initial correction
            t5_result = t5_corrector(text, max_length=512, do_sample=False)[0]['generated_text']
            
            # Use BART model for additional correction
            bart_result = bart_corrector(text, max_length=512, do_sample=False)[0]['generated_text']
            
            # Use advanced model if available
            if advanced_corrector:
                try:
                    advanced_result = advanced_corrector(text, max_new_tokens=512)[0]['generated_text']
                except Exception as e:
                    print(f"Error with advanced model: {str(e)}")
                    advanced_result = text
            else:
                advanced_result = text
            
            # Use LanguageTool for additional corrections
            language_tool_result = text
            matches = language_tool.check(text)
            if matches:
                language_tool_result = language_tool.correct(text)
            
            # Combine results - prefer advanced model if available, otherwise use T5
            if advanced_corrector:
                # Compare the results and choose the best one
                # For simplicity, we'll use the advanced model result
                final_result = advanced_result
            else:
                # If advanced model is not available, use T5 result
                final_result = t5_result
            
            # Apply LanguageTool corrections as a final step
            final_result = language_tool.correct(final_result)
            
            return final_result
        else:
            # Process each chunk separately and combine results
            corrected_chunks = []
            for chunk in chunks:
                # Use T5 model for correction
                corrected_chunk = t5_corrector(chunk, max_length=512, do_sample=False)[0]['generated_text']
                
                # Apply LanguageTool corrections
                matches = language_tool.check(corrected_chunk)
                if matches:
                    corrected_chunk = language_tool.correct(corrected_chunk)
                
                corrected_chunks.append(corrected_chunk)
            
            # Combine chunks
            final_result = " ".join(corrected_chunks)
            
            # Final pass with LanguageTool
            matches = language_tool.check(final_result)
            if matches:
                final_result = language_tool.correct(final_result)
            
            return final_result
    except Exception as e:
        print(f"Error in grammar correction: {str(e)}")
        # Return original text if correction fails
        return text

def text_to_speech(text, output_path):
    """Convert text to speech using pyttsx3"""
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path

def process_input(audio_file=None, text_input=None):
    """Main function to process audio or text input"""
    try:
        # Handle empty inputs
        if audio_file is None and not text_input:
            return {
                "error": "Please provide either an audio file or text input.",
                "transcription": "",
                "originalText": "",
                "correctedText": "",
                "grammarScore": 0
            }
        
        # Process audio input if provided
        transcript = ""
        if audio_file is not None:
            if not os.path.exists(audio_file):
                return {
                    "error": f"Audio file not found: {audio_file}",
                    "transcription": "",
                    "originalText": "",
                    "correctedText": "",
                    "grammarScore": 0
                }
            transcript = transcribe_audio(audio_file)
        
        # Use text input if provided, otherwise use transcript
        text_to_process = text_input if text_input else transcript
        
        if not text_to_process:
            return {
                "error": "No text to process.",
                "transcription": "",
                "originalText": "",
                "correctedText": "",
                "grammarScore": 0
            }
        
        # Correct grammar
        corrected_text = correct_grammar(text_to_process)
        
        # Extract features for grammar scoring
        if audio_file:
            # Combine audio and text features
            audio_feats = extract_audio_features(audio_file)
            text_feats = extract_text_features(text_to_process)
            features = {**audio_feats, **text_feats}
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
        grammar_score = float(grammar_model.predict(df)[0])
        
        return {
            "transcription": transcript,
            "originalText": text_to_process,
            "correctedText": corrected_text,
            "grammarScore": grammar_score
        }
    except Exception as e:
        print(f"Error in process_input: {str(e)}")
        return {
            "error": f"Error processing input: {str(e)}",
            "transcription": "",
            "originalText": "",
            "correctedText": "",
            "grammarScore": 0
        }

def validate_audio_file(file_path):
    """Validate audio file before processing"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            raise ValueError(f"File size ({file_size} bytes) exceeds maximum allowed size ({app.config['MAX_CONTENT_LENGTH']} bytes)")
        
        # Try to load the audio file with librosa
        try:
            y, sr = librosa.load(file_path, sr=None, duration=1)  # Load just first second to validate
        except Exception as e:
            raise ValueError(f"Invalid or unsupported audio file format: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Audio file validation failed: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = process_input(text_input=text)
    
    # Generate audio output
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{os.urandom(8).hex()}.wav")
    text_to_speech(result['correctedText'], audio_path)
    
    # Add audio path to result
    result['audioPath'] = f"/api/audio/{os.path.basename(audio_path)}"
    
    return jsonify(result)

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(file_path)
        
        # Validate the audio file
        try:
            validate_audio_file(file_path)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Invalid audio file: {str(e)}"}), 400
        
        # Process the audio
        result = process_input(audio_file=file_path)
        
        if 'error' in result:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(result), 400
        
        # Generate audio output
        try:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{os.urandom(8).hex()}.wav")
            text_to_speech(result['correctedText'], audio_path)
            result['audioPath'] = f"/api/audio/{os.path.basename(audio_path)}"
        except Exception as e:
            print(f"Error generating audio output: {str(e)}")
            result['audioPath'] = None
        
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in process_audio endpoint: {str(e)}")
        # Clean up any files that might have been created
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 