import os
import PIL
from PIL import Image
from skvideo.measure import strred
import numpy as np


if __name__ =='__main__':

    for i in range(6):
        fileset_name = 'Scene_'+str(i+1)
        if i==5:
            fileset_name = 'all_scenes'
        distorted_path = './'+fileset_name+'/output'
        reference_path = './'+fileset_name+'/ground_truth'

        distorted_file_list = os.listdir(distorted_path)
        reference_file_list = os.listdir(reference_path)

        distorted_pic_list = [np.array(Image.open(os.path.join(distorted_path, pic_name)).convert('L')) for pic_name in distorted_file_list]
        reference_pic_list = [np.array(Image.open(os.path.join(reference_path, pic_name)).convert('L')) for pic_name in reference_file_list]
        assert reference_pic_list[0].shape==distorted_pic_list[0].shape

        num_frame = min(len(distorted_pic_list), len(reference_pic_list))
        distorted_video = np.stack(distorted_pic_list[:num_frame])
        reference_video = np.stack(reference_pic_list[:num_frame])
        
        _ ,result, _ = strred(reference_video, distorted_video)
        
        print(fileset_name+f' STRRED is {result/2}')





