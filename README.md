# RAMOH_classify

## Getting Started
This project is implemented on ubuntu22.04 system and using python 3.9.

Clone the repo:
```
git clone git@github.com:juno5678/RAMOH_classify.git
```
Install the requirements using virtualenv or conda:

### pip
```
source scripts/install_pip.sh
```
### conda
```
source scripts/install_conda.sh
```
## Training and evaluation

Enter the command below to start training

```
python train.py --GT_path ./dataset/RAMOH_mds_updrs_label_20230516update_finish.xls -r ./dataset/video/ --output_dir ./result --display
```
