#!/bin/bash

# INSTALLATION
#sudo apt-get update
sudo apt install python3-pip
sudo apt install python3-tk
pip install virtualenv
ENV_DIR='atc_app'
virtualenv $ENV_DIR

$ENV_DIR/bin/pip install matplotlib==2.0.2
$ENV_DIR/bin/pip install numpy==1.13.3
$ENV_DIR/bin/pip install pandas==0.20.1
$ENV_DIR/bin/pip install scikit-learn==0.19.0
$ENV_DIR/bin/pip install scipy==1.0.1
$ENV_DIR/bin/pip install six==1.10.0

export PYTHONPATH=$PYTHONPATH:$(pwd)
$ENV_DIR/bin/python atc/single_app.py --flagfile=$1
