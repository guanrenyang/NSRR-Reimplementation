
import os

from numpy import pi, unicode_
from torch._C import set_flush_denormal
from torchvision.transforms.transforms import Resize
from base import BaseDataLoader

import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf

from PIL import Image

from typing import Union, Tuple, List

from utils import get_downscaled_size
import torchvision

class NSRRDataLoader(BaseDataLoader):
    """
    Generate batch of data
    `for x_batch in data_loader:`
    `x_batch` is a list of 4 tensors, meaning `view, depth, flow, view_truth`
    each size is (batch x channel x height x width)
    """
    def __init__(self,
                 data_dir: str,
                 img_dirname: str,
                 depth_dirname: str,
                 flow_dirname: str,
                 batch_size: int,
                 shuffle: bool = True,
                 validation_split: float = 0.0,
                 num_workers: int = 1,
                 downsample: Union[Tuple[int, int], List[int], int] = (2, 2),
                 num_data: Union[int,None] = None,
                 resize_factor : Union[int, None] = None,
                 ):
        dataset = NSRRDataset(data_dir,
                              img_dirname=img_dirname,
                              depth_dirname=depth_dirname,
                              flow_dirname=flow_dirname,
                              downsample=downsample,
                              num_data=num_data,
                              resize_factor = resize_factor
                              )
        super(NSRRDataLoader, self).__init__(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             validation_split=validation_split,
                                             num_workers=num_workers,
                                             )


class NSRRDataset(Dataset):
    """
    Requires that corresponding view, depth and motion frames share the same name.
    """
    def __init__(self,
                 data_dir: str,
                 img_dirname: str,
                 depth_dirname: str,
                 flow_dirname: str,
                 downsample: Union[Tuple[int, int], List[int], int] = (2, 2),
                 transform: nn.Module = None,
                 num_data:Union[int, None] = None,
                 resize_factor:Union[int, None] = None,
                 ):
        super(NSRRDataset, self).__init__()

        self.data_dir = data_dir
        self.img_dirname = img_dirname
        self.depth_dirname = depth_dirname
        self.flow_dirname = flow_dirname
        self.resize_factor = resize_factor

        if type(downsample) == int:
            downsample = (downsample, downsample)
        self.downsample = tuple(downsample)

        if transform is None:
            self.transform = tf.ToTensor()
        self.img_list = os.listdir(os.path.join(self.data_dir, self.img_dirname))
        self.img_list = sorted(self.img_list, key=lambda keys:[ord(i) for i in keys],reverse=False)
        
        self.data_list = []
        
        for i, img_name in enumerate(self.img_list):
            if(i>=num_data):
                break
            if(i==len(self.img_list)-4): # if current frame is the last img
                break   
            current_frame = self.img_list[i+4]
            pre_1, pre_2, pre_3, pre_4 = self.img_list[i+3], self.img_list[i+2], self.img_list[i+1], self.img_list[i]
            
            if 'a'<=current_frame[0]<='e' and not (current_frame[0]==pre_1[0]==pre_2[0]==pre_3[0]==pre_4[0]):
                continue
            else:
                self.data_list.append([current_frame, pre_1, pre_2, pre_3, pre_4])
                
    def __getitem__(self, index):
        # view
        # image_name = self.view_listdir[index]
        data = self.data_list[index]

        view_list, depth_list, flow_list, truth_list = [], [], [], []
        # elements in the lists following the order: current frame i, pre i-1, pre i-2, pre i-3, pre i-4
        for frame in data:
            img_path = os.path.join(self.data_dir, self.img_dirname, frame)
            depth_path = os.path.join(self.data_dir, self.depth_dirname, frame)
            flow_path = os.path.join(self.data_dir, self.flow_dirname, frame)
            
            img_view_truth = Image.open(img_path)
            img_flow = Image.open(flow_path)
            img_depth = Image.open(depth_path).convert(mode="L")

            img_view_truth = img_view_truth.resize((img_view_truth.size[0]//self.resize_factor, img_view_truth.size[1]//self.resize_factor), Image.ANTIALIAS)
            img_flow = img_flow.resize((img_flow.size[0]//self.resize_factor, img_flow.size[1]//self.resize_factor), Image.ANTIALIAS)
            img_depth = img_depth.resize((img_depth.size[0]//self.resize_factor, img_depth.size[1]//self.resize_factor), Image.ANTIALIAS)

            trans = self.transform

            img_view_truth = trans(img_view_truth)
            img_flow = trans(img_flow)

            downscaled_size = get_downscaled_size(img_view_truth.unsqueeze(0), self.downsample)

            trans_downscale = tf.Resize(downscaled_size)
            trans = tf.Compose([trans_downscale, trans])

            img_view = trans_downscale(img_view_truth)
            # depth data is in a single-channel image.
            img_depth = trans(img_depth)
            
            view_list.append(img_view)
            depth_list.append(img_depth)
            flow_list.append(img_flow)
            truth_list.append(img_view_truth)
            
        return view_list, depth_list, flow_list, truth_list[0]

    def __len__(self) -> int:
        return len(self.data_list)

