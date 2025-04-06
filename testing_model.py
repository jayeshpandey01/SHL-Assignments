import gradio as gr
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BitsAndBytesConfig
import language_tool_python
import pickle
import librosa
import numpy as np
import pandas as pd
import pyttsx3
import tempfile
import os

# Load Whisper model
whisper_model = whisper.load_model("base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load grammar correction model
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
correction_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
correction_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1").to(device)
corrector = pipeline("text2text-generation", model=correction_model, tokenizer=correction_tokenizer)

# Load final grammar scoring model
with open("outputs/grammar_scorer.pkl", "rb") as f:
    scorer_model = pickle.load(f)

# Grammar checker
tool = language_tool_python.LanguageTool('en-US')

# Audio feature extractor
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {
        "duration": librosa.get_duration(y=y, sr=sr),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "rmse": np.mean(librosa.feature.rms(y=y))
    }
    return features

# Text feature extractor
def extract_text_features(text):
    matches = tool.check(text)
    # We'll keep the error count for display purposes but won't include it in the model features
    return {
        "grammar_score_model": 0,  # Required by the model
        "num_words": len(text.split()),
        "avg_word_len": sum(len(w) for w in text.split()) / max(len(text.split()), 1),
        "text_length": len(text)
    }

# Combine features
def extract_combined_features(audio_path, transcript):
    audio_feats = extract_audio_features(audio_path)
    text_feats = extract_text_features(transcript)
    combined = {**audio_feats, **text_feats}
    df = pd.DataFrame([combined])
    return df

# TTS
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Main function
def process_input(audio=None, text_input=""):
    transcript = text_input
    tmp_path = None

    if audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        os.rename(audio, tmp_path)
        result = whisper_model.transcribe(tmp_path)
        transcript = result['text']

    # Grammar correction - handle longer texts
    try:
        # Split long text into smaller chunks if needed
        max_chunk_length = 500
        chunks = []
        
        # Simple chunking by sentences
        sentences = transcript.split('. ')
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
        
        # Process each chunk separately and combine results
        corrected_chunks = []
        for chunk in chunks:
            corrected_chunk = corrector(chunk, max_new_tokens=512)[0]['generated_text']
            corrected_chunks.append(corrected_chunk)
        
        # Combine chunks
        correction = " ".join(corrected_chunks)
    except Exception as e:
        print(f"Error in grammar correction: {str(e)}")
        correction = transcript  # Use original text if correction fails

    # Grammar scoring
    if tmp_path:
        features = extract_combined_features(tmp_path, transcript)
    else:
        text_feats = extract_text_features(transcript)
        dummy_audio_feats = {
            "duration": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid": 0.0,
            "rmse": 0.0
        }
        features = pd.DataFrame([{**dummy_audio_feats, **text_feats}])

    final_score = scorer_model.predict(features)[0]

    # TTS - save to file instead of speaking
    output_path = "corrected_output.wav"
    engine = pyttsx3.init()
    engine.save_to_file(correction, output_path)
    engine.runAndWait()

    return transcript, correction, final_score, output_path

# Gradio UI
demo = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Record or Upload Audio"),
        gr.Textbox(label="Or Enter Text Directly", placeholder="Type your text here...")
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Grammar-Corrected Text"),
        gr.Number(label="Predicted Grammar Score"),
        gr.Audio(label="Corrected Audio Output")
    ],
    title="Grammar Correction and Audio Processing",
    description="Upload an audio file or enter text to get grammar correction and audio output."
)

demo.launch()
