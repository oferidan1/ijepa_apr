import torch
from torchvision import transforms
from src.datasets.imagenet1k import make_imagenet1k
from pose.datasets.CameraPoseDataset import CambridgeDataset
from pose.util import utils
from src.superpoint import SuperPoint
import os
import numpy as np
import cv2
import pandas as pd
from os.path import join



def run_superpoint():
    device = 'cuda:0'
    checkpoint = 'checkpoint/superpoint_v1.pth.tar'
    nms_radius = 4
    keypoint_threshold = 0.005
    max_keypoints = 1024
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
    }
    superpoint = SuperPoint(config).to(device)
    superpoint.eval()
    
    root_path = '/dsi/scratch/home/dsi/rinaVeler/datasets/'
    image_folder = '.'
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    batch_size = 1
    
    #scene = 'Street' # One of these scenes: GreatCourt, Street, StMarysChurch, ShopFacade, OldHospital, KingsCollege
    out_path = root_path +'/superpoint/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)    

 # -- init data-loaders/samplers
    if data == 'ImageNet':
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,            
                root_path=root_path,
                image_folder=image_folder,
                copy_data=False)   
    else:
        
        transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
        
        dataset = CambridgeDataset(root_path, "pose/datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_Allscenes_train.csv", transform)
        
        loader_params = {'batch_size': batch_size,
                                    'shuffle': True,
                                    'num_workers': 8}
        unsupervised_loader = torch.utils.data.DataLoader(dataset, **loader_params)        
    
    for itr, (img, _, img_path) in enumerate(unsupervised_loader):
        img = img.to(device)
        with torch.no_grad():
            pred0 = superpoint({'image': img})
        file_name = img_path[0].split(os.sep)[-3]+'_'+img_path[0].split(os.sep)[-2]+'_'+os.path.splitext(os.path.basename(img_path[0]))[0]
        out_name  = out_path + file_name + '.npy'
        np.save(out_name, pred0)

def run_canny():
    labels_file = 'pose/datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_Allscenes_train.csv'
    df = pd.read_csv(labels_file)
    dataset_path = '/dsi/scratch/home/dsi/rinaVeler/datasets/'
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    out_path = dataset_path +'canny/'
    for path in imgs_paths:
        img = cv2.imread(path, 0)  # read image as grayscale
        resize_img = cv2.resize(img, (224,224))
        blurred_img = cv2.blur(resize_img, ksize=(5, 5))
        med_val = np.median(img)
        lower = int(max(0, 0.7 * med_val))
        upper = int(min(255, 1.3 * med_val))
        canny = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)
        #canny = cv2.Canny(img, 85, 255)
        filename = path.split(os.sep)[-3]+'_'+path.split(os.sep)[-2]+'_'+os.path.splitext(os.path.basename(path))[0]
        out_name  = out_path + filename + '.npy'
        np.save(out_name, canny)

            

if __name__ == '__main__':
    data = 'Cambridge'
    #run_superpoint()
    run_canny()
    