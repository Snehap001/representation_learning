#!/bin/bash

# Define variables
ENV_FILE="environment.yml"

# Ensure the environment.yml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found!"
    exit 1
fi

# Install dependencies using pip (as Conda is not available on Kaggle)
echo "Installing dependencies from environment.yml using pip..."
pip install -r <(awk '/^  - pip:/ {flag=1;next} /- / {flag=0} flag' $ENV_FILE)  # Extract packages listed under pip

echo "Environment setup completed successfully!"
