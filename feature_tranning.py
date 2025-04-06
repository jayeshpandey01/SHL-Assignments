import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load grammar correction model
print("Loading grammar model...")
tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1").to(device)
model.eval()

def extract_audio_features(audio_path):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(device)

        duration = waveform.size(1) / sr
        zcr = torch.mean(torch.abs(torch.diff(torch.sign(waveform)))).item()
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr).to(device)(waveform)
        spectral_centroid = torch.mean(mel_spec).item()
        rms = torch.sqrt(torch.mean(waveform**2)).item()

        return {
            "duration": duration,
            "zero_crossing_rate": zcr,
            "spectral_centroid": spectral_centroid,
            "rmse": rms,
        }

    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        return None

def extract_text_features(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return None

        input_text = "gec: " + text
        inputs = tokenizer([input_text], return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        original_words = text.strip().split()
        corrected_words = corrected_text.strip().split()
        total_words = max(len(original_words), 1)

        differences = sum(1 for o, c in zip(original_words, corrected_words) if o != c)
        diff_ratio = differences / total_words
        grammar_score = max(0, 1 - diff_ratio)  # Higher = better grammar

        return {
            "grammar_score_model": grammar_score,
            "original_text": text,
            "corrected_text": corrected_text,
            "num_words": total_words,
            "avg_word_len": sum(len(w) for w in original_words) / total_words,
            "text_length": len(text)
        }

    except Exception as e:
        print(f"Error extracting text features: {str(e)}")
        return None

def extract_combined_features(audio_path, transcript):
    audio_feats = extract_audio_features(audio_path)
    if audio_feats is None:
        return None

    text_feats = extract_text_features(transcript)
    if text_feats is None:
        return None

    return {**audio_feats, **text_feats}

def build_feature_dataframe(csv_path, audio_folder):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(audio_folder):
        raise FileNotFoundError(f"Audio folder not found: {audio_folder}")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Available columns in CSV: {df.columns.tolist()}")

    required_columns = ['filename', 'transcript', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    feature_rows = []
    print("Extracting features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        transcript = row['transcript']
        audio_path = os.path.join(audio_folder, filename)

        try:
            features = extract_combined_features(audio_path, transcript)
            if features is not None:
                features['filename'] = filename
                features['label'] = float(row['label'])
                feature_rows.append(features)
            else:
                print(f"Skipping {filename} due to feature extraction failure")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    if not feature_rows:
        raise ValueError("No features were successfully extracted")

    result_df = pd.DataFrame(feature_rows)
    print(f"✅ Successfully processed {len(result_df)} out of {len(df)} files")
    return result_df

if __name__ == "__main__":
    try:
        print("🔁 Starting feature extraction...")
        train_features_df = build_feature_dataframe("outputs/transcripts.csv", "dataset/audios_train")

        output_path = "outputs/hybrid_train_features.csv"
        train_features_df.to_csv(output_path, index=False)
        print(f"✅ Features saved to: {output_path}")

        print("\n📊 Feature Summary:")
        print(train_features_df.describe())

    except Exception as e:
        print(f"❌ Error during feature extraction: {str(e)}")
