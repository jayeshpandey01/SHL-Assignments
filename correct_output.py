import librosa
import numpy as np
import pandas as pd
import pyttsx3
import pickle
from transformers import pipeline
import gradio as gr
import whisper
import torch
import os

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# === Load Whisper Model for Transcription ===
whisper_model = whisper.load_model("base").to(device)

# === Load Hugging Face Grammar Correction Model ===
corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

# === Load Grammar Scoring Model ===
with open("output/grammar_scorer.pkl", "rb") as f:
    model = pickle.load(f)

# === Grammar Correction with LLM ===
def ai_grammar_correct(text):
    result = corrector(text, max_length=128, do_sample=False)[0]['generated_text']
    return result

# === Audio Feature Extractor ===
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return {
        "duration": librosa.get_duration(y=y, sr=sr),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "rmse": np.mean(librosa.feature.rms(y=y))
    }

# === Text Feature Extractor ===
def extract_text_features(text):
    words = text.split()
    return {
        "grammar_score_model": 0,  # Add this feature to match the model's expectations
        "num_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
        "text_length": len(text)
    }

# === Transcribe Audio with Whisper ===
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# === TTS function (using pyttsx3) ===
def speak_text_to_audio(text, output_path="corrected_output.wav"):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

# === Combine Audio and Text Features ===
def extract_combined_features(audio_path, transcript):
    audio_feats = extract_audio_features(audio_path)
    text_feats = extract_text_features(transcript)
    combined_feats = {**audio_feats, **text_feats}
    return combined_feats

# === Gradio Interface ===
def process_audio_and_text(audio_file, text_input):
    if audio_file is None and not text_input:
        return "Please provide either an audio file or text input.", None, None, None

    transcript = ""
    if audio_file is not None:
        # Transcribe audio using Whisper
        transcript = transcribe_audio(audio_file)

    if not text_input and not transcript:
        return "No text input provided.", None, None, None

    text_to_process = text_input if text_input else transcript
    corrected_text = ai_grammar_correct(text_to_process)
    
    # Extract features for scoring
    if audio_file:
        features = extract_combined_features(audio_file, text_to_process)
    else:
        # If no audio, use dummy audio features
        text_feats = extract_text_features(text_to_process)
        dummy_audio_feats = {
            "duration": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid": 0.0,
            "rmse": 0.0
        }
        features = {**dummy_audio_feats, **text_feats}
    
    # Predict grammar score
    df = pd.DataFrame([features])
    predicted_score = model.predict(df)[0]

    # Generate audio output
    speak_text_to_audio(corrected_text)

    return transcript, corrected_text, predicted_score, "corrected_output.wav"

# === Gradio Interface ===
iface = gr.Interface(
    fn=process_audio_and_text,
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

if __name__ == "__main__":
    iface.launch()
