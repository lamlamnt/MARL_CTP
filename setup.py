#!/bin/bash

# Exit script on error
set -e

# Set the virtual environment directory name
VENV_DIR=".venv"

# Step 1: Create the virtual environment
python3 -m venv "$VENV_DIR"

# Step 2: Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Step 3: Upgrade pip and install dependencies from requirements.txt
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
fi

# Step 4: Install the project in editable mode
pip install -e .