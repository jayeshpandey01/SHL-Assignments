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

<<<<<<< HEAD
- Python 3.8+
- Dependencies listed in `requirements.txt`
=======
### Flask Web Interface
![Flask Interface Output](images/image_02.png)
The Flask web interface provides a modern, user-friendly experience with:
- Clean audio recording interface
- Clear display of transcription, original text, and corrections
- Visual grammar score indicator
- Audio playback controls for corrected text

### Gradio Interface
![Gradio Interface Output](images/image_01.png)
The Gradio interface offers an alternative interface with:
- Simple audio recording and upload options
- Side-by-side display of original and corrected text
- Numerical grammar score display
- Downloadable corrected audio output

## Usage Examples

### Example 1: Audio Input
Input: Audio recording of the text:
"In the hot of the school year, the playground school is like a vibrant campus, a center-equal-full-jungle-gem stands or tall or normal with the climbing winds of lighter and ankles of excitement near by twins."

Output:
- **Transcription**: Accurate conversion of speech to text
- **Grammar Correction**: "In the hot of the school year, the playground school is like a vibrant campus, a center-equal-full-jungle-gem stands or tall or normal with the climbing winds of lighter and the ankles of excitement nearby twins."
- **Grammar Score**: 3.2/5.0
- **Audio Output**: Corrected text converted to speech

### Example 2: Text Input
Input: Text entry with grammar errors
Output:
- Original text preserved for comparison
- Grammar errors identified and corrected
- Grammar score calculated
- Option to hear corrected text via text-to-speech
>>>>>>> 104c8948aea91760e7d1442b000081c810983c44

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
4. Results will be saved to `dataset/grammar_results.csv` and `dataset/submission.csv`

## Output Format

The script generates two output files:

1. `dataset/grammar_results.csv` - Detailed results with the following columns:
   - `filename`: Original audio filename
   - `transcript`: Transcribed text from the audio
   - `grammar_score`: Grammar score (0-5)
   - `corrected_text`: Grammar-corrected version of the transcript

2. `dataset/submission.csv` - Submission format matching sample_submission.csv with:
   - `filename`: Original audio filename
   - `label`: Grammar score (0-5)

## Memory Management

The script includes memory management features to handle large audio files and prevent out-of-memory errors. It will automatically fall back to CPU if GPU memory is insufficient.

## Error Handling

<<<<<<< HEAD
The script includes comprehensive error handling to ensure that processing continues even if individual files fail. Errors are logged and included in the output CSV. 
=======
### Model Output Files (in `/outputs`)
- `grammar_scorer.pkl` - Trained grammar scoring model
- `hybrid_model.pkl` - Combined audio-text model
- Various CSV files containing training data and results

### Setup and Configuration
- `requirements.txt` - Python package dependencies
- `run.bat` and `run.sh` - Scripts to run the application on Windows and Unix systems
- `check_model.py` - Validates model existence and creates dummy model if needed
- `test_server.py` - Server testing utilities

### Documentation
- `README.md` - Project documentation and setup instructions
- `TROUBLESHOOTING.md` - Troubleshooting guide for common issues

### Directories
- `/templates` - Contains web interface templates
- `/uploads` - Temporary storage for uploaded audio files
- `/outputs` - Stores model files and training results
- `/dataset` - Contains training data
- `/models` - Stores pre-trained models
- `/offload_folder` - Used for model memory optimization

## Technical Details

### Memory Management
- Automatic memory cleanup for GPU operations
- Fallback to CPU when GPU memory is insufficient
- 4-bit quantization for models to reduce memory usage

### Error Handling
- Comprehensive error handling throughout the application
- Graceful fallbacks when operations fail
- Detailed error logging for debugging

### Text Processing
- Chunking mechanism for handling long texts
- Sentence-based processing for better grammar correction
- Multiple model approach for improved accuracy

## Troubleshooting

If you encounter any issues, please refer to the `TROUBLESHOOTING.md` file for common solutions. For connection issues, try:

1. Ensure the Flask server is running
2. Check if port 8080 is available
3. Run `python test_server.py` to diagnose connection issues
4. Check your firewall settings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Whisper for audio transcription
- Hugging Face for grammar correction models
- LanguageTool for additional grammar checking 
>>>>>>> 104c8948aea91760e7d1442b000081c810983c44
