# Information for replicating the system in your lab

## Hardware preparation
A multicamera recording system and a PC for data processing are needed, as shown below.

### The multicamera system
Any system that can capture video from multiple cameras in sync should work. Precise synchronization (~1 ms) of video frames from different cameras is critical for accurate 3D reconstruction and tracking. For examples, we have used [Motif video recording system](http://loopbio.com/recording/) and [StreamPix](https://www.norpix.com/products/streampix/streampix.php). 

The cameras must be fixed stably because the information about the camera positions is needed for the 3D reconstruction and the camera positions are assumed to remain unchanged throughout the recording. In addition, if the cameras are located inside the cage like us, appropriate camera housings are needed.

See [our paper](xxx) for an example of implementation. 

### The data processing PC
A Linux or Windows PC with a NVIDIA GPU is required. Multiple GPUs can be utilized by providing different GPU names (as [*device_str*](run_demo.py) variables) for each process.

### Budget consideration
Costs of materials used in our study were:
- A Motif video recording system (8 cameras): 5M yen.
- A data processing PC with 2 GPUs: 1M yen.
- Custom camera housings and their install: 2.5M yen. 

## Software preparation
See [getting_started.md](getting_started.md) for information on installing software for the data processing. 

## Camera calibaration
Camera parameters need be calibrated and saved in the [calib folder](calib). Specifically, 

- Camera position (tvec) and pose (rvec) in *cam_extrinsic_optim.h5*
- Camera intrinsic parameters (K, xi, D) in *cam_intrinsic.h5*
- Camera IDs and image size (width, height) in *config.yaml*

See [OpenCV documentation](https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html) for details of parameters.

## Training of the CNNs
The pipeline uses three CNNs, the pose estimation, ID recognition, and detection networks. Training of the CNNs fine-tuned to your experimental environment is important for good performance. The training datasets for our macaque group cage are available [here](XXX). Create similar annotated datasets based on the images captured in your experimental setup. For better performance with a relatively small custom dataset, it may be good to try combining it with our larger dataset or re-training our [pre-trained networks](XXX) with your dataset.

For details of the dataset formats and how to train the CNNs, see OpenMMLab documentations (https://github.com/open-mmlab). *MMDetection*, *MMPose*, and *MMClassification* frameworks are used for training the CNNs. The model configuration files for each network can be found in the [model folder](model). After training, replace the pth files in the weight folder with the new ones. 

## Analyzing your own data
The present pipeline support [IMGStore](https://github.com/loopbio/imgstore) format for the input data. Put the input data in the videos folder, specify the data name as the *data_name* variable in [run_demo.py](run_demo.py), and run the script.

The resultant numerical data of the 3D motion is saved in *./results3D/[data_name]/kp3d.pickle*. In the pickle file, *kp3d* variable is a 4D numpy matrix of [Animals] * [Video frames] * [Keypoints] * [X,Y,Z], representing keypoint positions. *kp3d_score* variable is a 3D numpy matrix of [Animals] * [Video frames] * [Keypoints], representing the confidence score of each keypoint estimation. 

## Processing time:
In our PC using a single Quadro RTX 8000 GPU, the whole processing took around 30 min for the 1-min demo data. You can also try [running the demo](getting_started.md) with your own data processing PC. 

## Others:
For questions, feel free to open an issue in this github repository or to email to us (Jumpei Matsumoto or Takaaki Kaneko).





