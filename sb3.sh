#!/bin/sh

echo "1. Install virtual environment"
sudo apt install python3-venv

echo "2. Create virtual environment"
python3 -m venv sb3

echo "3. Activate virtual environment"
source sb3/bin/activate

echo "4. Install requirements"
pip install -r sb3-requirements.txt