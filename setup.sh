#!/bin/bash
# Install required packages
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! Don't forget to:"
echo "1. Copy example.env to .env: cp example.env .env"
echo "2. Add your API keys to .env file"