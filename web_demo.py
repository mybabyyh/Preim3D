# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import gradio as gr
import numpy as np
import dnnlib
import time
# import legacy
import torch
import glob
import os, sys
import cv2
from torch_utils import misc
# from renderer import Renderer
# from training.networks import Generator
from utils.model_utils import setup_model
from utils.common import tensor2im
# from utils.alignment import align_face
from PIL import Image
from editings import latent_editor
import torchvision.transforms as transforms
from camera_utils import LookAtPoseSampler, GaussianCameraPoseSampler

device = torch.device('cuda')
port   = int(sys.argv[1]) if len(sys.argv) > 1 else 21111

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_camera_traj(generator, pitch, yaw):
    camera_lookat_point = torch.tensor(generator.rendering_kwargs['avg_camera_pivot'], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2 + yaw, 3.14 / 2 + pitch, camera_lookat_point,
                                              radius=generator.rendering_kwargs['avg_camera_radius'],
                                              device=device)
    focal_length = 4.2647
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                              device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    return c


def check_name(network_pkl=None):
    """Gets model by name."""
    if network_pkl is None or  network_pkl == 'FFHQ512':
        network_pkl = "./pretrained/preim3d_ffhq.pt"
    generator_pkl = './pretrained/eg3d_G_ema.pkl'

    # if model_name == 'FFHQ512':
    #
    # # elif model_name == 'FFHQ256v1':
    # #     network_pkl = "./pretrained/ffhq_256v1.pkl"
    # else:
    #     if os.path.isdir(model_name):
    #         network_pkl = sorted(glob.glob(model_name + '/*.pkl'))[-1]
    #     else:
    #         network_pkl = model_name
    return network_pkl, generator_pkl


def get_model(network_pkl, generator_name):
    print('Loading networks from "%s"...' % network_pkl)
    net, opts = setup_model(network_pkl, generator_name, device)
    generator = net.decoder
    generator.eval()
    print('compile and go through the initial image')
    init_cam = get_camera_traj(generator, 0, 0)
    z_samples = np.random.RandomState(123).randn(1, 512)
    # sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
    w_samples = generator.mapping(z=torch.from_numpy(z_samples).to(device), c=init_cam, truncation_psi=0.7,
                                  truncation_cutoff=14)
    dummy = generator.synthesis(w_samples, init_cam, noise_mode='const')['image']
    res = dummy.shape[-1]
    imgs = np.zeros((res, res, 3))

    return generator, net, res, imgs


print(check_name())
global_states = list(get_model(check_name()[0], check_name()[1]))
wss  = [None]

transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


n_age = torch.load(f'./editings/interfacegan_directions/inversion/Young.pt').to(device).reshape(14, 512)
n_smile = torch.load(f'./editings/interfacegan_directions/inversion/Smiling.pt').to(device).reshape(14, 512)
n_blackhair = torch.load(f'./editings/interfacegan_directions/inversion/Black_Hair.pt').to(device).reshape(14, 512)
n_eyeglasses = torch.load(f'./editings/interfacegan_directions/inversion/Eyeglasses.pt').to(device).reshape(14, 512)
n_goatee = torch.load(f'./editings/interfacegan_directions/inversion/Goatee.pt').to(device).reshape(14, 512)
n_lipstick = torch.load(f'./editings/interfacegan_directions/inversion/Wearing_Lipstick.pt').to(device).reshape(14, 512)
n_makeup = torch.load(f'./editings/interfacegan_directions/inversion/Heavy_Makeup.pt').to(device).reshape(14, 512)
n_male = torch.load(f'./editings/interfacegan_directions/inversion/Male.pt').to(device).reshape(14, 512)
n_wavyhair = torch.load(f'./editings/interfacegan_directions/inversion/Wavy_Hair.pt').to(device).reshape(14, 512)

def proc_seed(history, seed):
    if isinstance(seed, str):
        seed = 0
    else:
        seed = int(seed)

def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def f_synthesis(yaw, pitch, age, smile, eyeglasses, lipstick, blackhair, goatee, input_image=None):

    model_name="./pretrain3d/preim3d_ffhq.pt"
    model_find='' 
    input_image = input_image.convert('RGB')
    input_image = transform(input_image).to(device)
    history = gr.get_state() or {}
    seeds = []
    if model_find != "":
        model_name = model_find

    model_name, generator_name = check_name(model_name)
    if model_name != history.get("model_name", None):
        model, net, res, imgs = get_model(model_name, generator_name)
        global_states[0] = model
        global_states[1] = net
        global_states[2] = res
        global_states[3] = imgs

    model, net, res, imgs = global_states
    history['model_name'] = model_name
    history['generator_name'] = generator_name
    gr.set_state(history)
    set_random_seed(sum(seeds))

    latent = get_latents(net, input_image.unsqueeze(0))
    print('latent:', latent.shape)
    ws = latent.clone()

    # editing  age, smile, eyeglasses, lipstick, blackhair, goatee
    delta_ws = -age * n_age + smile * n_smile + eyeglasses * n_eyeglasses \
               + lipstick * n_lipstick + blackhair * n_blackhair + goatee * n_goatee
    ws_edit = ws + delta_ws

    cam = get_camera_traj(model, pitch, yaw)
    # ws = torch.cat([ws, ws], dim=0)
    # cam = torch.cat([cam, cam], dim=0)

    start_t = time.time()

    with torch.no_grad():
        image_inv = model.synthesis(ws, c=cam, noise_mode='const')['image']
        image_edit = model.synthesis(ws_edit, c=cam, noise_mode='const')['image']

    end_t = time.time()
    image_inv = image_inv[0].permute(1,2,0).detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
    image_edit = image_edit[0].permute(1,2,0).detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

    if image_inv.shape[0] == image_inv.shape[0]:
        image = np.concatenate([image_inv, image_edit], 1)
    else:
        a = image_edit.shape[0]
        b = int(image_inv.shape[1] / image_inv.shape[0] * a)
        print(f'resize {a} {b} {image_edit.shape} {image_inv.shape}')
        image = np.concatenate([cv2.resize(image_inv, (b, a), cv2.INTER_AREA), image_edit], 1)
  
    print(f'rendering time = {end_t-start_t:.4f}s')
    return (image * 255).astype('uint8')


input_image = gr.inputs.Image(type='pil', label="input_image")
yaw    = gr.inputs.Slider(minimum=-0.8, maximum=0.8, default=0, label="yaw")
pitch  = gr.inputs.Slider(minimum=-0.6, maximum=0.6, default=0, label="pitch")
age    = gr.inputs.Slider(minimum=-10, maximum=10, default=0, label="age")
smile    = gr.inputs.Slider(minimum=-10, maximum=10, default=0, label="smile")
eyeglasses    = gr.inputs.Slider(minimum=-10, maximum=10, default=0, label="eyeglasses")
lipstick    = gr.inputs.Slider(minimum=-10, maximum=10, default=0, label="lipstick")
blackhair    = gr.inputs.Slider(minimum=-10, maximum=10, default=0, label="blackhair")
goatee    = gr.inputs.Slider(minimum=-10, maximum=10, default=0, label="goatee")


css = ".output_image {height: 40rem !important; width: 100% !important;}"
gr.Interface(fn=f_synthesis,
             inputs=[yaw, pitch, age, smile, eyeglasses, lipstick, blackhair, goatee, input_image],
             outputs="image",
             layout='unaligned',
             css=css,
             live=True,
             server_name='localhost',
             server_port=port).launch(share=True)
