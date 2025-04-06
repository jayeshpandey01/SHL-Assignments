# Troubleshooting Guide

## Connection Issues

### "This site can't be reached" Error

If you see an error like "This site can't be reached" or "ERR_CONNECTION_REFUSED" when trying to access the application, try the following steps:

1. **Check if the server is running**
   - Make sure you've started the application by running `python app.py` or using the provided scripts
   - Look for output in the terminal indicating that the server has started

2. **Try a different port**
   - The application now uses port 8080 by default
   - If that doesn't work, try accessing http://localhost:5000, http://localhost:3000, or http://localhost:8000

3. **Run the test script**
   - Run `python test_server.py` to check if the server is accessible
   - This script will try different ports and provide diagnostic information

4. **Check firewall settings**
   - Your firewall might be blocking the connection
   - Temporarily disable your firewall to see if that resolves the issue
   - If it does, add an exception for Python in your firewall settings

5. **Check for other running servers**
   - Another application might be using the same port
   - Try closing other applications that might be using port 8080
   - You can also modify the port in app.py to use a different one

## Model Loading Issues

### "No module named 'whisper'" Error

If you see an error about missing modules, make sure you've installed all the required dependencies:

```
pip install -r requirements.txt
```

### "Could not load grammar_scorer.pkl" Error

If the application can't find the grammar scoring model:

1. Run the check_model.py script to create a dummy model:
   ```
   python check_model.py
   ```

2. Make sure the outputs directory exists and contains the grammar_scorer.pkl file

### Advanced Grammar Model Loading Issues

If you encounter errors related to the advanced grammar correction models:

1. **Memory Issues**
   - The advanced models require significant memory
   - Try closing other applications to free up memory
   - If you have a GPU, make sure it has enough VRAM

2. **Model Download Issues**
   - The application will automatically download models on first run
   - Check your internet connection
   - If downloads fail, you can manually download the models:
     ```
     python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('prithivida/grammar_error_correcter_v1'); AutoModelForSeq2SeqLM.from_pretrained('prithivida/grammar_error_correcter_v1')"
     ```

3. **Fallback to Basic Models**
   - If the advanced model fails to load, the application will automatically fall back to basic models
   - You'll still get grammar correction, but it may be less accurate
   - Check the console output for messages about model loading

4. **LanguageTool Issues**
   - LanguageTool requires Java to be installed
   - If you see errors related to LanguageTool, install Java:
     - Windows: Download and install from https://www.java.com/download/
     - Linux: `sudo apt install default-jre`
     - Mac: `brew install java`

## Audio Recording Issues

### "Could not access microphone" Error

If you see an error about accessing the microphone:

1. Make sure your browser has permission to access the microphone
2. Check your system settings to ensure the microphone is enabled
3. Try using a different browser

### Audio File Upload Issues

If you're having trouble uploading audio files:

1. **File size too large**
   - The application has a 16MB limit for file uploads
   - Try compressing your audio file or using a shorter recording

2. **Unsupported file format**
   - The application supports common audio formats like WAV, MP3, M4A
   - If your file isn't being processed, try converting it to a supported format

3. **Upload fails**
   - Check your internet connection
   - Try refreshing the page and uploading again
   - Make sure the file isn't corrupted

4. **Processing takes too long**
   - Larger files take longer to process
   - The application uses AI models that require significant processing power
   - For faster results, try using shorter audio clips or text input instead

## Still Having Issues?

If you're still experiencing problems:

1. Check the terminal output for error messages
2. Try running the application with debug mode enabled:
   ```
   python app.py
   ```
3. Look for any error messages in the browser's developer console (F12)
4. Contact support with the specific error message you're seeing 