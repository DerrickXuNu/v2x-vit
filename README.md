# V2X-ViT
This is the official implementation of ECCV2022 paper "V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer".

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

## Data
### Download
The data can be found from [google url](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6?usp=sharing).  Since the data for train/validate/test
is very large, we  split each data set into small chunks, which can be found in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:

```
cat train.zip.parta* > train.zip
unzip train.zip
```
### Structure
After downloading is finished, please make the file structured as following:

```sh
v2x-vit # root of your OpenCOOD
├── v2xset # the downloaded v2xset data
│   ├── train
│   ├── validate
│   ├── test
├── v2xvit # the core codebase

```
### Details
Our data label format is very similar with the one in [OPV2V](https://github.com/DerrickXuNu/OpenCOOD). For more details, please refer to the [data tutorial](docs/data_intro.md).

### Noise Simulation
One important feature of V2XSet is the capability of adding different communication noises. This is done in a post-processing approach through our flexible coding framework. To set different noise, please
refer to [config yaml tutorial](docs/config_tutorial.md).

