import torch
from torchvision import transforms
from src.datasets.imagenet1k import make_imagenet1k
from src.superpoint import SuperPoint
import os
import numpy as np

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
    
    root_path = 'F:/imagenet/'
    image_folder = '.'
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor()])
    batch_size = 1
    
    out_path = root_path + 'superpoint/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)    

 # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,            
            root_path=root_path,
            image_folder=image_folder,
            copy_data=False)           
    
    for itr, (img, masks_enc, img_path) in enumerate(unsupervised_loader):
        img = img.to(device)
        with torch.no_grad():
            pred0 = superpoint({'image': img})
        file_name = os.path.splitext(os.path.basename(img_path[0]))[0]
        out_name  = out_path + file_name + '.npy'
        np.save(out_name, pred0)          
            

if __name__ == '__main__':
    run_superpoint()
    