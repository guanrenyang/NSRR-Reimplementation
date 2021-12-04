from os import curdir
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch

# from base import BaseModel


class FeatureExtraction(nn.Module):
    '''Feature Extraction Network (特征提取网络)'''
    def __init__(self, kernel_size = 3, padding = 'same'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels= 4, out_channels= 32, kernel_size= kernel_size, padding= padding),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= kernel_size, padding= padding),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 8, kernel_size= kernel_size, padding= padding),
            nn.ReLU()
        )
    def forward(self, color_map: torch.Tensor, depth_map: torch.Tensor, channel_dim = 1)->torch.Tensor: # `channel`is in dim_1 by default
        # color map: (3, Height, Width)
        # depth map: (1, Height, Width)
        x = torch.concat((color_map, depth_map), dim= channel_dim) 
        h = self.net(x)
        return torch.concat((h, x), dim= channel_dim)

class FeatureReweighting(nn.Module):
    '''Feature Reweighting Network (特征重加权网络))'''
    def __init__(self, kernel_size = 3, padding = 'same', scale = 10):
        super().__init__()

        self.scale = scale

        self.net = nn.Sequential(
            # We think of the input as the concatanation of RGB-Ds of 5 frames, each of which has 4 channles
            # so `in_channels=20`, which is 4*5
            nn.Conv2d(in_channels= 20, out_channels= 32, kernel_size= kernel_size, padding= padding),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= kernel_size, padding= padding),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 4, kernel_size= kernel_size, padding= padding),
            nn.Tanh()
        )
    def forward(self, upsampled_current_feature: torch.Tensor, previous_features: List[torch.Tensor], channel_dim = 1)->List[torch.Tensor]:
        # each previous feature has 4 channels 
        # 4 previous frames in all
        assert previous_features[0].shape[1]==4
        x = torch.concat((upsampled_current_feature,)+tuple(previous_features), dim= channel_dim)
        w = self.net(x)
        w = (w-(-1))/2*10 # Scale
        weighted_previous_features = [w[:,i,:,:]*previous_features[i] for i in range(4)] # Reweighting
        return weighted_previous_features
    
class Reconstruction(nn.Module):
    '''Recomstriction Network (重建网络)'''
    def __init__(self, kernel_size=3, padding='same'):
        super().__init__()
        self.pooling = nn.MaxPool2d(2)

        self.num_previous_frames = 4

        # Split the network into 5 groups of 2 layers to apply concat operation at each stage
        # todo: the first layer of the model would take
        # the concatenated features of all previous frames,
        # so the input number of channels of the first 2D convolution
        # would be 12 * self.number_previous_frames
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(12*self.num_previous_frames+12, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.center = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),

        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(32+64, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )


    def forward(self, current_feature: torch.Tensor, previous_features: List[torch.Tensor], channel_dim=1) -> torch.Tensor:
        # Features of the current frame and the reweighted features
        # of previous frames are concatenated
        x = torch.cat((current_feature,)+ tuple(previous_features), channel_dim)

   
        out_encoder_1 = self.pooling(self.encoder_1(x))
        
        out_encoder_2 = self.pooling(self.encoder_2(out_encoder_1))
        
        out_center = self.center(out_encoder_2)
        
        out_decoder_1 = self.decoder_1(torch.concat((out_center, out_encoder_2), dim= channel_dim))

        out_decoder_2 = self.decoder_2(torch.concat((out_encoder_1, out_decoder_1), dim= channel_dim))

        return out_decoder_2

net_1 = FeatureExtraction()
x = torch.ones((1, 3, 8 ,16))
y = torch.ones((1, 1, 8 ,16))
z = torch.ones((1, 4, 8 ,16))
# print(x.shape[1])
# print(torch.concat((x,)+(y,z), dim = 1).shape)
# print(net(x, y).shape)
# print(torch.concat((x,y), dim=1).shape)

net_2 = FeatureReweighting()
current = torch.ones((1, 4, 8, 16))
previous = [torch.ones((1, 4, 8, 16)) for i in range(4)]
# for i in range(4):
#     print(net_2(current, previous)[i].shape)

net_3 = Reconstruction()
current = torch.ones((1, 12, 8, 16))
previous = [torch.ones((1, 12, 8, 16)) for i in range(4)]
print(net_3(current, previous).shape)
        
