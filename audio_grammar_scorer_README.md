# Audio Grammar Scorer

A simple tool that takes audio files as input, transcribes them using Whisper, and outputs grammar scores in label form using a pre-trained model.

## Features

- Transcribes audio files using Whisper
- Extracts audio and text features
- Calculates grammar scores using a pre-trained model
- Supports both single file and directory processing
- Memory-efficient processing with GPU support

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure the grammar scoring model is available at `outputs/grammar_scorer.pkl`

## Usage

### Process a Single Audio File

```
python audio_grammar_scorer.py --input path/to/audio_file.wav --output result.csv
```

### Process All Audio Files in a Directory

```
python audio_grammar_scorer.py --input path/to/audio_directory --output results.csv --directory
```

### Command Line Arguments

- `--input`, `-i`: Input audio file or directory (required)
- `--output`, `-o`: Output CSV file (default: grammar_scores.csv)
- `--directory`, `-d`: Process all audio files in the input directory

## Output Format

The output CSV file contains the following columns:
- `filename`: Original audio filename
- `transcript`: Transcribed text from the audio
- `label`: Grammar score (0-5)

## Memory Management

The script includes memory management features to handle large audio files and prevent out-of-memory errors. It will automatically fall back to CPU if GPU memory is insufficient.

## Error Handling

The script includes comprehensive error handling to ensure that processing continues even if individual files fail. Errors are logged and included in the output CSV. 