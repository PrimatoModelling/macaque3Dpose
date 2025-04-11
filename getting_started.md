# Getting started 
This document explains how to run the demo.

# Additional dataset for the demo
Please download the video data and pretrained models from [Zenodo](XXXXX).   
Place the unzipped contents ("videos" and "weight" folders) in the same directory as this file. 

# Tested environment 
Ubuntu 20.04, NVIDIA Driver Version: 450.102.04

# Python version 
3.8.19

# Installing tools
```bash
cd $PathForDownloadedFolder

# pytorch 
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# Openmmlab
pip install openmim==0.3.9
mim install mmcv-full==1.6.2
mim install mmdet==2.26.0
mim install mmpose==0.29.0
mim install mmtrack==0.14.0
mim install mmcls==0.25.0
pip install xtcocotools==1.12 # needed to be downgraded due to compatibility to numpy

# Major tools
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python==4.8.1.78
pip install numba==0.58.0
pip install h5py==3.9.0
pip install pyyaml==6.0.1
pip install toml==0.10.2 
pip install matplotlib==3.7.5
pip install joblib==1.3.2
pip install networkx==2.6

# Minor or local resources 
pip install imgstore==0.2.9
cd src/m_lib/
python setup.py build_ext --inplace
```

# Run demo 
```bash 
cd $PathForDownloadedFolder

python run_demo.py # the results will appear at "./output"
```