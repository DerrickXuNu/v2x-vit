# V2X-ViT
This is the official implementation of ECCV2022 paper "V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer". The code and data will be released by middle of July. 

## Installation
```bash
# Clone repo
git clone https://github.com/DerrickXuNu/v2x-vit

cd v2x-vit

# Setup conda environment
conda create -y --name v2xvit python=3.7

conda activate v2xvit
# pytorch >= 1.8.1, newest version can work well
conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
# spconv 2.0 install, choose the correct cuda version for you
pip install spconv-cu113

# Install dependencies
pip install -r requirements.txt
# Install bbx nms calculation cuda version
python v2xvit/utils/setup.py build_ext --inplace

# install v2xvit into the environment
python setup.py develop
```
