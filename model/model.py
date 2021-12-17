from os import curdir
from typing import List, Tuple
from warnings import filterwarnings
from torch._C import set_flush_denormal
from torch.cuda import current_blas_handle
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from base import BaseModel

from typing import Union, List, Tuple, Callable, Any

from pytorch_colors import rgb_to_hsv, rgb_to_ycbcr, ycbcr_to_rgb
from utils import upsample_zero_2d
# from base import BaseModel
class NSRR(BaseModel):
    def __init__(self, upsample_scale, height=10, length=10) -> None:
        super().__init__()

        self.cur_i_fea_ext  = FeatureExtraction()
        self.pre_i1_fea_ext = FeatureExtraction()
        self.pre_i2_fea_ext = FeatureExtraction()
        self.pre_i3_fea_ext = FeatureExtraction()
        self.pre_i4_fea_ext = FeatureExtraction()

        # self.cur_i_upsample_1 = ZeroUpSampling(upsample_scale)
        # self.cur_i_upsample_2 = ZeroUpSampling(upsample_scale)
        # self.pre_i1_upsample = ZeroUpSampling(upsample_scale)
        # self.pre_i2_upsample = ZeroUpSampling(upsample_scale)
        # self.pre_i3_upsample = ZeroUpSampling(upsample_scale)
        # self.pre_i4_upsample = ZeroUpSampling(upsample_scale)
        self.zero_upsample = ZeroUpSampling(upsample_scale)
        self.backward_warp = BackwardWarp()

        self.feature_reweighting = FeatureReweighting()
        self.reconstruction = Reconstruction()


    def forward(self, view_list, depth_list, flow_list, channel_dim = 1):
        '''
        view size: (batch, channel=3, height_low, width_low)
        depth size: (batch, channel=1, height_low, width_low)
        flow size: (batch, channle=3, height_high, width_high)
        truth size: (batch, channel=3, height_high, width_high)
        '''
        cur_i_view, cur_i_depth, cur_i_flow = view_list[0], depth_list[0], flow_list[0]
        pre_i1_view, pre_i1_depth, pre_i1_flow = view_list[1], depth_list[1], flow_list[1]
        pre_i2_view, pre_i2_depth, pre_i2_flow = view_list[2], depth_list[2], flow_list[2]
        pre_i3_view, pre_i3_depth, pre_i3_flow = view_list[3], depth_list[3], flow_list[3]
        pre_i4_view, pre_i4_depth, pre_i4_flow = view_list[4], depth_list[4], flow_list[4]

        # current frame path 1
        h_cur_i_1 = self.zero_upsample(self.cur_i_fea_ext(rgb_to_ycbcr(cur_i_view), cur_i_depth))
        # current frame path 2
        h_cur_i_2 = self.zero_upsample(torch.concat((cur_i_view, cur_i_depth), dim=channel_dim))
        
        # zero upsample for previous frames
        pre_i1_sampled = self.zero_upsample(self.pre_i1_fea_ext(pre_i1_view, pre_i1_depth))
        pre_i2_sampled = self.zero_upsample(self.pre_i2_fea_ext(pre_i2_view, pre_i2_depth))
        pre_i3_sampled = self.zero_upsample(self.pre_i3_fea_ext(pre_i3_view, pre_i3_depth))
        pre_i4_sampled = self.zero_upsample(self.pre_i4_fea_ext(pre_i4_view, pre_i4_depth))
        
        # backward warp
        '''
        Commented by RenyangGuan
        We are not sure to use the i-1th motion to warp the i-1th frame 
            or to use the ith motion to warp the i-1th frame 
        '''
        pre_i1_warped = self.backward_warp(pre_i1_sampled, pre_i1_flow)

        _ = self.backward_warp(pre_i2_sampled, pre_i2_flow)
        pre_i2_warped = self.backward_warp(_, pre_i1_flow)

        _ = self.backward_warp(pre_i3_sampled, pre_i3_flow)
        _ = self.backward_warp(_, pre_i2_flow)
        pre_i3_warped = self.backward_warp(_, pre_i1_flow)

        _ = self.backward_warp(pre_i4_sampled, pre_i4_flow)
        _ = self.backward_warp(_, pre_i3_flow)
        _ = self.backward_warp(_, pre_i2_flow)
        pre_i4_warped = self.backward_warp(_, pre_i1_flow)

        # Feature reweighting
        previous_feature_list = [pre_i1_warped, pre_i2_warped, pre_i3_warped, pre_i4_warped]
        weighted_previous_feature_list = self.feature_reweighting(h_cur_i_2, previous_feature_list)
        out = self.reconstruction(h_cur_i_1, weighted_previous_feature_list)
        
        return ycbcr_to_rgb(out)
        

class FeatureExtraction(BaseModel):
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
        # output channel 0-7: trained features
        # output channel 8-10: raw color map
        # output channel 11: raw depth map
        x = torch.concat((color_map, depth_map), dim= channel_dim) 
        h = self.net(x)
    
        return torch.concat((h, x), dim= channel_dim)

class ZeroUpSampling(BaseModel):
    """
    Basic layer for zero-upsampling of 2D images (4D tensors).
    """

    scale_factor: Tuple[int, int]

    def __init__(self, scale_factor: Union[Tuple[int, int], List[int], int]):
        super(ZeroUpSampling, self).__init__()
        if type(scale_factor) == int:
            scale_factor = (scale_factor, scale_factor)
        assert(len(scale_factor) == 2)
        self.scale_factor = tuple(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample_zero_2d(x, scale_factor=self.scale_factor)

class BackwardWarp(BaseModel):
    """
    A model for backward warping 2D image tensors according to motion tensors.
    """
    def __init__(self):
        super(BackwardWarp, self).__init__()

    def forward(self, x_image: torch.Tensor, x_flow: torch.Tensor=None) -> torch.Tensor:
        # see: https://discuss.pytorch.org/t/image-warping-for-backward-flow-using-forward-flow-matrix-optical-flow/99298
        # input image is: [batch, channel, height, width]
        # if x_image.shape[2:] != x_flow.shape[2:]

        x_motion = self.optical_flow_to_motion(x_flow)

        index_batch, _, height, width = x_image.size()
        grid_x = torch.arange(width).view(1, -1).repeat(height, 1)
        grid_y = torch.arange(height).view(-1, 1).repeat(1, width)
        grid_x = grid_x.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
        grid_y = grid_y.view(1, 1, height, width).repeat(index_batch, 1, 1, 1)
        ##
        grid = torch.cat((grid_x, grid_y), 1).float()

        # grid is: [batch, channel (2), height, width]
        vgrid = grid + x_motion
        # Grid values must be normalised positions in [-1, 1]
        vgrid_x = vgrid[:, 0, :, :]
        vgrid_y = vgrid[:, 1, :, :]
        vgrid[:, 0, :, :] = (vgrid_x / width) * 2.0 - 1.0
        vgrid[:, 1, :, :] = (vgrid_y / height) * 2.0 - 1.0
        # swapping grid dimensions in order to match the input of grid_sample.
        # that is: [batch, output_height, output_width, grid_pos (2)]
        vgrid = vgrid.permute((0, 2, 3, 1))
        output = F.grid_sample(x_image, vgrid, mode='bilinear', align_corners=False)
        return output
    
    def optical_flow_to_motion(self, rgb_flow: torch.Tensor, sensitivity: float = 0.5) -> torch.Tensor:
        """
        Returns motion vectors as a [batch, 2, height, width]
        with [:, 0, :, :] the abscissa and [:, 1, :, :] the ordinate.
        """
        # flow is: batch x 3-channel x height x width
        # todo: rgb_to_hsv  is extremely slow (from pytorch_color)
        # around 300ms on my machine
        # with [1, 3, 1080, 1920] tensor: single 1920 x 1080 image...
        hsv_flow = rgb_to_hsv(rgb_flow)
        motion_length = hsv_flow[:, 2, :, :] / sensitivity
        motion_angle = (hsv_flow[:, 0, :, :] - 0.5) * (2.0 * np.pi)
        motion_x = - motion_length * torch.cos(motion_angle)
        motion_y = - motion_length * torch.sin(motion_angle)
        motion_x.unsqueeze_(1)
        motion_y.unsqueeze_(1)
        motion = torch.cat((motion_x, motion_y), dim=1)
        # motion is: batch x 2-channel x height x width
        return motion

class FeatureReweighting(BaseModel):
    '''Feature Reweighting Network (特征重加权网络))'''
    def __init__(self, kernel_size = 3, padding = 'same', scale = 10):
        super().__init__()

        self.scale = scale

        self.net = nn.Sequential(
            # We think of the input as the concatanation of RGB-Ds of current frame, which has 4 channles
            # and full features of previous frames, each of which has 12 channels
            # so `in_channels=20`, which is 4+4*12 = 52
            nn.Conv2d(in_channels= 52, out_channels= 32, kernel_size= kernel_size, padding= padding),
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
    
class Reconstruction(BaseModel):
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

class LayerOutputModelDecorator(BaseModel):
    """
    A Decorator for a Model to output the output from an arbitrary set of layers.
    """

    def __init__(self, model: nn.Module, layer_predicate: Callable[[str, nn.Module], bool]):
        super(LayerOutputModelDecorator, self).__init__()
        self.model = model
        self.layer_predicate = layer_predicate

        self.output_layers = []

        def _layer_forward_func(layer_index: int) -> Callable[[nn.Module, Any, Any], None]:
            def _layer_hook(module_: nn.Module, input_, output) -> None:
                self.output_layers[layer_index] = output
            return _layer_hook
        self.layer_forward_func = _layer_forward_func

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(
                    self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(torch.Tensor())

    def forward(self, x) -> List[torch.Tensor]:
        self.model(x)
        return self.output_layers

   # For Debug     
# net_1 = FeatureExtraction()
# x = torch.ones((1, 3, 8 ,16))
# y = torch.ones((1, 1, 8 ,16))
# z = torch.ones((1, 4, 8 ,16))
# # print(x.shape[1])
# # print(torch.concat((x,)+(y,z), dim = 1).shape)
# # print(net(x, y).shape)
# # print(torch.concat((x,y), dim=1).shape)

# net_2 = FeatureReweighting()
# current = torch.ones((1, 4, 8, 16))
# previous = [torch.ones((1, 4, 8, 16)) for i in range(4)]
# # for i in range(4):
# #     print(net_2(current, previous)[i].shape)

# net_3 = Reconstruction()
# current = torch.ones((1, 12, 8, 16))
# previous = [torch.ones((1, 12, 8, 16)) for i in range(4)]
# # print(net_3(current, previous).shape)

# net_4 = BackwardWarp()
# current = torch.ones((1, 12, 8, 16))
# img = cv2.imread('T:\\GitHub\\NSRR-Reimplementation\\model\\motion.png')
# img_tensor = torch.tensor(img).reshape((1, 3, 480, 720))
# print(img_tensor.shape)
# img_tensor_rgb = (net_4.optical_flow_to_motion(img_tensor)).view((-1, 480, 720))
# img_rgb = np.array(img_tensor_rgb)
# print(img_tensor_rgb.shape)
# cv2.imshow('Image', img_rgb)
# input()
