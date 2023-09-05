import torch
import sys
sys.path.append(".")
sys.path.append("..")
from editings import ganspace, sefa
from utils.common import tensor2im
from camera_utils import LookAtPoseSampler


class LatentEditor(object):
    def __init__(self, eg3d_generator, is_cars=False):
        self.generator = eg3d_generator
        self.is_cars = is_cars 

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return self._latents_to_image(edit_latents)

    def apply_interfacegan(self, latent, c, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent
            for i in range(len(direction)):
                edit_latents += factor[i] * direction[i]
        return self._latents_to_image(edit_latents, c), edit_latents

    def apply_sefa(self, latent, indices=[2, 3, 4, 5], **kwargs):
        edit_latents = sefa.edit(self.generator, latent, indices, **kwargs)
        return self._latents_to_image(edit_latents)

    # Currently, in order to apply StyleFlow editings, one should run inference,
    # save the latent codes and load them form the official StyleFlow repository.
    # def apply_styleflow(self):
    #     pass

    def _latents_to_image(self, latents, c):
        with torch.no_grad():
            # images, _ = self.generator([latents], randomize_noise=False, input_is_latent=True)
            ws = latents
            images = self.generator.synthesis(ws, c, noise_mode='const')['image']
            if self.is_cars:
                images = images[:, :, 64:448, :]  # 512x512 -> 384x512
        # horizontal_concat_image = torch.cat(list(images), 2)
        # final_image = tensor2im(horizontal_concat_image)
        # return final_image
        return images
