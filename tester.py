# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train import main as app_main
from src.models.vision_transformer import vit_huge
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--checkpoint', type=str, default='checkpoint/IN1K-vit.h.14-300e.pth.tar',
    help='checkpoint to load for testing')

# Print the layers/modules of the model for inspection
def print_model_layers(model, prefix=""):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Module):
            module_name = prefix + "." + name if prefix else name
            print(module_name)
            print_model_layers(module, prefix=module_name)
            
def load_backbone(load_path):    

    # Initialize the ViT-H model with the specified patch size and resolution
    encoder = vit_huge(patch_size=14, num_classes=1000)  # Adjust num_classes if needed
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = ckpt['encoder']

    # -- loading encoder
    for k, v in pretrained_dict.items():
        encoder.state_dict()[k[len("module."):]].copy_(v)
    
    encoder.to('cuda:0')
    print_model_layers(encoder)
    print("done!")

if __name__ == '__main__':
    args = parser.parse_args()
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    load_backbone(args.checkpoint)

        