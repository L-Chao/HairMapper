import os.path
import argparse
import cv2
from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from tqdm import tqdm
from classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from mapper.networks.level_mapper import LevelMapper
import torch
import glob
from diffuse.inverter_remove_hair import InverterRemoveHair
import numpy as np
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

from encoder4editing.encode import ImageEncoder

class HairMapper(object):
    def __init__(self,
                 mapper_model: str,
                 parsing_net_model: str) -> None:
        model_name = 'stylegan2_ada'
        latent_space_type = 'wp'
        truncation_psi = 0.75

        print(f'Initializing generator.')
        model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=truncation_psi)


        mapper = LevelMapper(input_dim=512).eval().cuda()
        ckpt = torch.load(mapper_model)
        alpha = float(ckpt['alpha']) * 1.2
        mapper.load_state_dict(ckpt['state_dict'], strict=True)
        kwargs = {'latent_space_type': latent_space_type}
        parsing_net = get_parsingNet(save_pth=parsing_net_model)

        inverter = InverterRemoveHair(
            model_name,
            Generator=model,
            learning_rate=0.01,
            reconstruction_loss_weight=1.0,
            perceptual_loss_weight=5e-5,
            truncation_psi=truncation_psi,
            logger=None)

        self.alpha = alpha
        self.kwargs = kwargs
        self.parsing_net = parsing_net
        self.inverter = inverter
        self.model = model
        self.dilate_kernel_size = 50
        self.blur_kernel_size = 50
        self.mapper = mapper
    
    def inference(self,
                  latent_codes_origin: np.ndarray,
                  use_defuse: bool,
                  origin_img_path: str,
                  res_save_path: str) -> None:
        mapper_input = latent_codes_origin.copy()
        mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
        edited_latent_codes = latent_codes_origin
        edited_latent_codes[:, :8, :] += self.alpha * self.mapper(mapper_input_tensor).to('cpu').detach().numpy()

        origin_img = cv2.imread(origin_img_path)

        outputs = self.model.easy_style_mixing(latent_codes=edited_latent_codes,
                                          style_range=range(7, 18),
                                          style_codes=latent_codes_origin,
                                          mix_ratio=0.8,
                                          **self.kwargs
                                          )
        edited_img = outputs['image'][0][:, :, ::-1]

        hair_mask = get_hair_mask(img_path=origin_img, net=self.parsing_net, include_hat=True, include_ear=False)
        mask_dilate = cv2.dilate(hair_mask,
                                 kernel=np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(self.blur_kernel_size, self.blur_kernel_size))
        mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)
        
        face_mask = 255 - mask_dilate_blur

        index = np.where(face_mask > 0)
        cy = (np.min(index[0]) + np.max(index[0])) // 2
        cx = (np.min(index[1]) + np.max(index[1])) // 2
        center = (cx, cy)

        if use_defuse:
            synthesis_image = origin_img * (1 - hair_mask // 255) + edited_img * (hair_mask // 255)

            target_image = (synthesis_image[:, :, ::-1]).astype(np.uint8)
            res_wp, _, res_img = self.inverter.easy_mask_diffuse(target=target_image,
                                                            init_code=edited_latent_codes,
                                                            mask=hair_mask, iteration=150)

            # Image Blending in Sec 3.7
            mixed_clone = cv2.seamlessClone(origin_img, res_img[:, :, ::-1], face_mask[:, :, 0], center,
                                            cv2.NORMAL_CLONE)
        else:

            mixed_clone = cv2.seamlessClone(origin_img, edited_img, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)
        cv2.imwrite(res_save_path, mixed_clone)


if __name__ == '__main__':
    # run()
    encode_model_path = '/home/luanchao/HairMapper/ckpts/e4e_ffhq_encode.pt'
    image_encoder = ImageEncoder(encode_model_path)
    
    mapper_model = './mapper/checkpoints/final/best_model.pt'
    parsing_net_model = './ckpts/face_parsing.pth'
    hair_mapper =  HairMapper(mapper_model, parsing_net_model)

    image_path = './test_data/origin/jb.jpg'
    output_path = './test_data/origin/mapper_jb.jpg'
    code = image_encoder.inference(image_path)
    hair_mapper.inference(code, False, image_path, output_path)
    # np.save('code.npy', code)
