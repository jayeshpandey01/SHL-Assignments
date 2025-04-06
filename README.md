# Batch Grammar Assistant

This tool processes audio files listed in a CSV file, transcribes them, calculates grammar scores, and outputs the results to a new CSV file.

## Features

- Transcribes audio files using Whisper
- Extracts audio and text features
- Calculates grammar scores
- Corrects grammar in transcribed text
- Processes multiple files in batch
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

## Usage

1. Place your audio files in the `dataset/audios_test` directory
2. Ensure your `test.csv` file is in the `dataset` directory with a column named "filename" containing the audio filenames
3. Run the script:
   ```
   python batch_grammar_assistant.py
   ```
4. Results will be saved to `dataset/grammar_results.csv`

## Output Format

The output CSV file contains the following columns:
- `filename`: Original audio filename
- `transcript`: Transcribed text from the audio
- `grammar_score`: Grammar score (0-5)
- `corrected_text`: Grammar-corrected version of the transcript

## Memory Management

The script includes memory management features to handle large audio files and prevent out-of-memory errors. It will automatically fall back to CPU if GPU memory is insufficient.

## Error Handling

The script includes comprehensive error handling to ensure that processing continues even if individual files fail. Errors are logged and included in the output CSV. 