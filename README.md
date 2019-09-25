# Citywalks

The repo contains the Citywalks dataset and code for the paper: 

Multiple Object Forecasting: Predicting Future Object Locations in Diverse Environments. WACV 2020 (To appear).

Currently, this repo contains the Citywalks dataset, raw tracking labels, ground truth trajectory labels, STED trajectory predictions, and evaluation code. More content, including the the code for STED, will be added soon.

Citywalks contains a total of 501 20-second video clips of which 358 contain at least one valid pedestrian trajectory.  

# Downloading Citywalks videos
The raw videos can be dowloaded here: [[Google Drive](https://drive.google.com/open?id=1oMN-fsWvEjUZ9Ah_3JwUuIY7cmR0OP_Q)]
 
It may be difficult to use wget or curl to download files from Google Drive. We recommend using gdown if you wish to download files using the terminal:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=FILE_ID
```

Where FILE_ID is the ID of the file in the Google Drive URL.

# Downloading tracking results

We use Yolov3 and Mask-RCNN to detect pedestrians, and DeepSORT to track pedestrians. Tracking results can be downloaded here: [[Google Drive](https://drive.google.com/open?id=12-_FiphR5m0Yd455pem13OVnvCvi-yIn)]

# Evaluating Multiple Object Forecasting models

We use ADE, FDE, AIOU, and FIOU metrics to evaluate Multiple Object Forecasting models on Citywalks. The ground truth trajectories and predictions from STED can be downloaded here: [[Google Drive](https://drive.google.com/open?id=1KZ08pzp1j8P598VNIMR3vSeFcnf3871d)]

These predictions can then be evaluated using the file ```evaluate_outputs.py```. Example usage:

```bash
python evaluate_outputs.py -gt ./outputs/ground_truth/ -pred ./outputs/sted/
``` 


Please open an issue if you have any questions.
