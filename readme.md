# macaque3Dpose 

<img src="./imgs/MovieS1.gif" width="800">　  

This tool is designed to process multiple videos from various viewpoints and generate 3D pose estimations for groups of macaques. It is especially useful for analyzing the behavior of freely moving macaques in laboratory environments. The tool can be applied in areas such as biomedical, ethological, applied animal sciences and neuroscience research. 

- Analytic pipeline combining multiple CNNs and 3D reconstruction utilities. 
- Pretrained network optimized for 3D pose estimation of macaque groups. The pre-trained model outputs time series data of 3D positions for nose, ears, shoulders, elbows, wrists, hips, knees, ankles. 
- The training dataset is available [here](XXX). 

### Reference 
If you use the code or data, please cite us:   
[Three-dimensional markerless motion capture of multiple freely behaving monkeys for automated characterization of social behavior](xxx)   
by Jumpei Matsumoto and Takaaki Kaneko et al. ([journal of xxx](xxx))

### Demo 
Please see [getting_started.md](getting_started.md)  

### Info for replication in your lab
Please see [info_replication.md](info_replication.md)

### License
- The tools and data are made available under the MIT license.   
- This repository incorporates third-party materials (anipose and mvpose). See [Third Party Notices](./ThirdPartyNotices.txt) for more information.  

### Acknowledgment 
Our work builds on the previous significant contributions:
- [anipose](https://github.com/lambdaloop/anipose)  
  Karashchuk, P., Rupp, K.L., Dickinson, E.S., Walling-Bell, S., Sanders, E., Azim, E., Brunton, B.W., and Tuthill, J.C. (2021). Anipose: A toolkit for robust markerless 3D pose estimation. Cell Rep. 36, 109730.
- [mvpose](https://github.com/zju3dv/mvpose)  
  Dong, J., Jiang, W., Huang, Q., Bao, H., and Zhou, X. (2019). Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views. arXiv:1901.04111.
- [openmmlab](https://github.com/open-mmlab)  
  Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al. (2019). MMDetection: Open MMLab Detection Toolbox and Benchmark. arXiv:1906.07155.

**Funding**:
This work was supported by MEXT/JSPS KAKENHI Grant Numbers 16H06534, 19H05467, 22H05157, 22K07325, 22K19480, and 23H02781, by JST Grant Number JPMJMS2295-12 and JPMJFR2320, by the Takeda Science Foundation, and by the National Institutes of Natural Sciences (NINS) Joint Research Grant Number 01111901.

## Ethical considerations
All procedures for the use and experiments of Japanese macaques were approved by the Animal Welfare and Animal Care Committee of the Center for the Evolutionally Origins of the Human Behavior, Kyoto University, followed by the Guidelines for Care and Use of Nonhuman Primates established by the same institution.

