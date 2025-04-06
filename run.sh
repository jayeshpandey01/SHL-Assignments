#!/bin/bash

echo "Setting up Grammar Assistant..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7+ and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Check if model exists, create if not
echo "Checking for grammar model..."
python3 check_model.py

# Run the application
echo "Starting Grammar Assistant..."
python3 app.py 