import torch
from torchvision import transforms
from src.datasets.imagenet1k import make_imagenet1k
from src.superpoint import SuperPoint
import os
import numpy as np
import cv2

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

def run_canny():
    filename = "frame00001.jpg"
    img = cv2.imread(filename, 0)  # read image as grayscale
    blurred_img = cv2.blur(img, ksize=(5, 5))
    med_val = np.median(img)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    canny = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
    #canny = cv2.Canny(img, 85, 255)
    cv2.imwrite('image1.png', canny)



if __name__ == '__main__':
    #run_superpoint()
    run_canny()
    