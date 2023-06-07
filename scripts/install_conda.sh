#!/usr/bin/env bash

export CONDA_ENV_NAME=romoh-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.9

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install -r requirements.txt
