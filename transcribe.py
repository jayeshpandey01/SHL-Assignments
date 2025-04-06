import numpy as np
import os
import whisper
import pandas as pd
from tqdm import tqdm
import torch
import glob

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Whisper model with GPU support if available
model = whisper.load_model("medium").to(DEVICE)

def process_audio_folder(audio_folder, output_file):
    results = []
    errors = []
    
    # Make the path dynamic
    base_path = "C:/Users/HARSHAL/Downloads/SHL Research Intern Assignment/dataset"
    folder_path = os.path.join(base_path, 'audios_train')
    
    # Get all WAV files in the folder
    audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(f"\nFound {len(audio_files)} audio files in {audio_folder}")
    print(f"Folder path: {folder_path}")
    
    if not audio_files:
        print(f"No .wav files found in {folder_path}")
        return
    
    print("\nFirst few files found:")
    for f in audio_files[:5]:
        print(f"- {f}")
    
    for audio_path in tqdm(audio_files, desc=f"Transcribing {audio_folder}"):
        audio_path = os.path.normpath(audio_path)
        
        if not os.path.exists(audio_path):
            error_msg = f"File not found: {audio_path}"
            print(error_msg)
            errors.append(error_msg)
            continue
            
        try:
            result = model.transcribe(audio_path, language="en", task="transcribe", fp16=DEVICE=="cuda")
            file_name = os.path.basename(audio_path)
            results.append({
                'file_name': file_name,
                'transcription': result['text']
            })
        except Exception as e:
            error_msg = f"Error transcribing {audio_path}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    # Save transcription results
    if results:
        output_path = os.path.join(base_path, output_file)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\nTranscription complete. Results saved to {output_path}")
        print(f"Successfully transcribed {len(results)} files")
    else:
        print(f"\nNo files were successfully transcribed in {audio_folder}")
    
    # Save error log if there were any errors
    if errors:
        error_log_path = os.path.join(base_path, f"error_log_{audio_folder}.txt")
        with open(error_log_path, 'w') as f:
            f.write('\n'.join(errors))
        print(f"Errors have been logged to {error_log_path}")


try:
    # Process the `train` folder
    process_audio_folder("train", "train_transcriptions_direct.csv")
finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
