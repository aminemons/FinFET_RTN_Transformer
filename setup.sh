#!/bin/bash
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements (including PyTorch with CUDA 12.1)..."
pip install -r requirements.txt

echo "Setup complete. You can now run the training script:"
echo "python -m src.training.train --batch_size 256 --num_workers 32"
