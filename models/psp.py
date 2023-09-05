import matplotlib

matplotlib.use('Agg')
import os
import numpy as np
import imageio
import torch
from torch import nn
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from camera_utils import LookAtPoseSampler
import math

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.stylegan_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'WPlusEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'UNetEncoder':
            encoder = psp_encoders.UNetEncoder(self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading preim3d over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            # self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            # self.__load_latent_avg(ckpt)
            if self.opts.train_decoder:
                print('train decoder!!!!')
                self.decoder = torch.load(os.path.abspath(self.opts.eg3d_generator)).requires_grad_(True).to(self.opts.device)
            else:
                self.decoder = torch.load(os.path.abspath(self.opts.eg3d_generator)).to(self.opts.device)
            self.__load_latent_avg(repeat=self.encoder.style_count)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(os.path.abspath(model_paths['ir_se50']))
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            # ckpt = torch.load(self.opts.eg3d_generator)
            # self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            # self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)
            if self.opts.train_decoder:
                self.decoder = torch.load(os.path.abspath(self.opts.eg3d_generator)).requires_grad_(True).to(self.opts.device)
            else:
                self.decoder = torch.load(os.path.abspath(self.opts.eg3d_generator)).to(self.opts.device)
            self.__load_latent_avg(repeat=self.encoder.style_count)

    def forward(self, x, c, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code

        images = self.decoder.synthesis(codes, c, noise_mode='const')['image'] # warm up
        result_latent = codes

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, repeat=None):
        z_samples = np.random.RandomState(123).randn(10000, self.decoder.z_dim)
        camera_lookat_point = torch.tensor(self.decoder.rendering_kwargs['avg_camera_pivot'], device=self.opts.device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=self.decoder.rendering_kwargs['avg_camera_radius'],
                                                  device=self.opts.device)
        focal_length = 4.2647
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.opts.device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c = c.repeat(len(z_samples), 1)
        w_samples = self.decoder.mapping(z=torch.from_numpy(z_samples).to(self.opts.device), c=c, truncation_psi=0.7, truncation_cutoff=14)
        self.latent_avg = w_samples.mean(0, keepdim=True)
