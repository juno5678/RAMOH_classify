#!/usr/bin/env bash

echo "Creating virtual environment"
python3.9 -m venv romoh-env
echo "Activating virtual environment"

source $PWD/romoh-env/bin/activate

$PWD/romoh-env/bin/pip install -r requirements.txt
