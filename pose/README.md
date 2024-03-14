## I-JEPA-APR
This repository implements I-JEPA-APR


### Setup

1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset
1. Setup a conda env:
```
conda create -n loc python=3.9
pip install torch torchvision
pip install scikit-image
pip install pandas
conda activate loc
```

---
download i-jepa checkpoint from https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
copy to 'checkpoint'
### Usage

The entry point for training and testing is the main.py script in the root directory

  For detailed explanation of the options run:
  ```
  python main_apr.py -h
  ```
  For example, in order to train TransPoseNet on the ShopFacade scene from the CambridgeLandmarks dataset: 
  ```
python main_apr.py --mode=train --dataset_path=<path to the CambridgeLandmarks dataset> --labels_file ./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
  ```
  Your checkpoints (.pth file saved based on the number you specify in the configuration file) and log file
  will be saved under an 'out' folder.
  
  In order to test your model, for example on the the ShopFacade scene:
  ```
python main_apr.py --mode=test --dataset_path=<path to the CambridgeLandmarks dataset> --test_labels_file ./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_test.csv --checkpoint_path <path to .pth>
  ```
 
 download ijepa pre-trained model and put in checkpoint: https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
