import torch
import torch.nn.functional as F
import torch.nn as nn
from src.models.vision_transformer import vit_huge

def load_backbone(load_path):    
    # Initialize the ViT-H model with the specified patch size and resolution
    encoder = vit_huge(patch_size=14, num_classes=1000)  # Adjust num_classes if needed
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = ckpt['encoder']

    # -- loading encoder
    for k, v in pretrained_dict.items():
        encoder.state_dict()[k[len("module."):]].copy_(v)
            
    return encoder

class PoseRegressor(nn.Module):
    """
    A class to represent camera pose regressor
    """
    def __init__(self, backbone_path, config):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(PoseRegressor, self).__init__()
        self.backbone = load_backbone(backbone_path)
        backbone_dim = 1280
        latent_dim = 1024
        self.head_x = nn.Sequential(nn.Linear(backbone_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 3))                
        self.head_q = nn.Sequential(nn.Linear(backbone_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 4))     
        
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d((1,backbone_dim))          
        
        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight) 
        

    def forward(self, data):
        x = self.backbone(data['img'])
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        p_x = self.head_x(x)
        p_q = self.head_q(x)
        return {'pose': torch.cat((p_x, p_q), dim=1)}
    
    def forward_backbone(self, data):
        x = self.backbone(data['img'])
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)        
        return x
    
    def forward_heads(self, x):
        p_x = self.head_x(x)
        p_q = self.head_q(x)
        return {'pose': torch.cat((p_x, p_q), dim=1)}
        
          