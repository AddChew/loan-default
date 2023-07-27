#!/bin/bash

conda create -n loan-default python=3.10 -y
source activate loan-default
pip install -r requirements.txt