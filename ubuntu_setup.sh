#!/bin/sh
sudo apt-get update -y

echo "1. Install pip"
sudo apt install python3-pip

echo "2. Install mysql dependencies"
sudo apt-get install -y default-libmysqlclient-dev
sudo apt-get install libmysqlclient-dev

echo "3. Install python packages"
pip install -r requirements.txt