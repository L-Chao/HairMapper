from argparse import Namespace
import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
from PIL import ImageFile
import glob
import os
import argparse
sys.path.append(".")
sys.path.append("..")
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.psp import pSp

def run_on_batch(inputs, net):
    latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return latents

class ImageEncoder(object):
    def __init__(self, 
                 model_path: str) -> None:
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts= Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        self.net = net
        self.img_transforms = transforms.Compose([
           transforms.Resize((256, 256)),
           transforms.ToTensor(),
           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def inference(self, file_path: str) -> np.ndarray:
        input_image = PIL.Image.open(file_path)
        transformed_image = self.img_transforms(input_image)
        with torch.inference_mode():
            latents = run_on_batch(transformed_image.unsqueeze(0), self.net)
            latent = latents[0].cpu().numpy()
            latent = np.reshape(latent,(1,18,512))
        return latent

if __name__ == '__main__':
    run()

