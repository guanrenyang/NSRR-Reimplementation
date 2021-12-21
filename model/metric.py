import torch
import numpy
import math
import pytorch_ssim

def psnr(img1:torch.Tensor, img2: torch.Tensor):
    assert img1.shape == img2.shape
    img1=img1.cpu().detach().numpy()
    img2=img2.cpu().detach().numpy()
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1:torch.Tensor, img2:torch.Tensor):
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

    return pytorch_ssim.ssim(img2, img1)

