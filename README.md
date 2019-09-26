# Multiple Object Forecasting

The repo contains the Citywalks dataset and code for the paper: 

Multiple Object Forecasting: Predicting Future Object Locations in Diverse Environments. WACV 2020 (to appear).

Currently, this repo contains the Citywalks dataset, raw tracking labels, ground truth trajectory labels, STED trajectory predictions, and evaluation code. More content, including the code for STED, will be added soon.

Citywalks contains a total of 501 20-second video clips of which 358 contain at least one valid pedestrian trajectory.  

# Downloading Citywalks videos
The raw videos can be downloaded here: [[Google Drive](https://drive.google.com/open?id=1oMN-fsWvEjUZ9Ah_3JwUuIY7cmR0OP_Q)] 
[[One Drive](https://1drv.ms/u/s!AlCsPcYF4WdzjGVQzrAPeteUH4rS?e=tET0GY)]

To download from the terminal:
```bash
pip install gdown
gdown https://drive.google.com/uc?id==1oMN-fsWvEjUZ9Ah_3JwUuIY7cmR0OP_Q
```
OR 
```bash
wget -O citywalks.zip "https://onedrive.live.com/download?cid=7367E105C63DAC50&resid=7367E105C63DAC50%211637&authkey=AJmkgXYpBLsX-CM"
```

# Downloading tracking results

We use Yolov3 and Mask-RCNN to detect pedestrians, and DeepSORT to track pedestrians. Tracking results can be downloaded here: [[Google Drive](https://drive.google.com/open?id=12-_FiphR5m0Yd455pem13OVnvCvi-yIn)] [[One Drive](https://onedrive.live.com/?authkey=%21AJvssSoCKvoJ2ew&cid=7367E105C63DAC50&id=7367E105C63DAC50%211635&parId=root&action=locate)]

To download from the terminal:
```bash
gdown https://drive.google.com/uc?id=12-_FiphR5m0Yd455pem13OVnvCvi-yIn
```
OR 
```bash
wget -O tracks.zip "https://onedrive.live.com/download?cid=7367E105C63DAC50&resid=7367E105C63DAC50%211635&authkey=AOlVtKFP76Rhn90"
```

The files are organized as follows:

- vid: Name of video clip
- filename: City of original recording
- frame_num: Frame number. 30FPS.
- track: Track ID assigned by DeepSORT
- cx: Center x bounding box coordinate
- cy: Center y bounding box coordinate
- w: Bounding box width
- h: Bounding box height
- track_length: Current length (in frames) of this track
- labelled: 1 if this frame is labelled, 0 otherwise. For a track to be labelled, it must follow at least 29 previous tracked frames and have at least 60 following tracked frames. i.e. the pedestrian must have been tracked continuously for at least 3 seconds.
- requires_features: 1 if this frame requires features, 0 otherwise. All labelled frames and the previous 29 frames require features. This is the motion history used for MOF.


# Evaluating Multiple Object Forecasting models

We use ADE, FDE, AIOU, and FIOU metrics to evaluate Multiple Object Forecasting models on Citywalks. The ground truth trajectories and predictions from STED can be downloaded here: [[Google Drive](https://drive.google.com/open?id=1KZ08pzp1j8P598VNIMR3vSeFcnf3871d)] [[One Drive](https://1drv.ms/u/s!AlCsPcYF4WdzjGTxFjuPdOkhb-4p?e=3nylt1)]

To download from the terminal:
```bash
gdown https://drive.google.com/uc?id=1KZ08pzp1j8P598VNIMR3vSeFcnf3871d
```
OR 
```bash
wget -O outputs.zip "https://onedrive.live.com/download?cid=7367E105C63DAC50&resid=7367E105C63DAC50%211636&authkey=AFNJWoXVjzfuNVI"
```

These predictions can then be evaluated using the file ```evaluate_outputs.py```. Example usage:

```bash
python evaluate_outputs.py -gt ./outputs/ground_truth/ -pred ./outputs/sted/
``` 

If run correctly, the result should be equal to Table 2 in our paper.

The files are organized as follows:
- vid: Name of video clip
- filename: City of original recording
- frame_num: Frame number. 30FPS.
- x1_t: Left bounding box coordinate at t timesteps in the future
- y1_t: Top bounding box coordinate at t timesteps in the future
- x2_t: Right bounding box coordinate at t timesteps in the future
- y2_t: Bottom bounding box coordinate at t timesteps in the future

Please open an issue if you have any questions.

# Acknowledgements
We would also like to thank Daniel Sczepansky for collecting the videos used in the Citywalks dataset. Full videos are available at: https://www.youtube.com/poptravelorg
