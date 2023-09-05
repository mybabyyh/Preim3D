import argparse
import random

import torch
import numpy as np
import sys
import os
import imageio
import scipy.interpolate
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, dir_path)
sys.path.insert(0, parent_dir_path)

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from PIL import Image
from editings import latent_editor
from camera_utils import LookAtPoseSampler, GaussianCameraPoseSampler
import json
from utils import common
import math


def main(args):
    net, opts = setup_model(args.ckpt, args.eg3d_generator, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    editor = latent_editor.LatentEditor(net.decoder, is_cars)

    edit_direction_dir = args.edit_direction_dir

    # initial inversion
    # latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

    # set the editing operation
    if args.edit_attribute == 'inversion':
        pass
    else:
        interfacegan_directions = {
                'age': f'{edit_direction_dir}/Young.pt',
                'smile': f'{edit_direction_dir}/Smiling.pt',
                'eyeglass': f'{edit_direction_dir}/Eyeglasses.pt',
                'goatee': f'{edit_direction_dir}/Goatee.pt',
                'male': f'{edit_direction_dir}/Male.pt',
                'wavyhair': f'{edit_direction_dir}/Wavy_Hair.pt',
                'grayhair': f'{edit_direction_dir}/Gray_Hair.pt',
                'lipstick': f'{edit_direction_dir}/Wearing_Lipstick.pt',
                'makeup': f'{edit_direction_dir}/Heavy_Makeup.pt',
                'blackhair': f'{edit_direction_dir}/Black_Hair.pt',
                'brownhair': f'{edit_direction_dir}/Brown_Hair.pt',
                'mouthopen': f'{edit_direction_dir}/Mouth_Slightly_Open.pt'
        }

        edit_attributes = args.edit_attribute.split(',')
        edit_degrees = [float(d) for d in args.edit_degree.split(',')]
        assert len(edit_attributes) == len(edit_degrees)
        assert args.batch == 1

        edit_direction = [torch.load(interfacegan_directions[att]).to(device).reshape(14, 512) for att in edit_attributes]

    edit_directory_path = os.path.join(args.save_dir, args.edit_attribute)
    os.makedirs(edit_directory_path, exist_ok=True)

    # perform high-fidelity inversion or editing
    processed_count = 0
    for batch, file_name in data_loader:
        if args.n_sample is not None and processed_count > args.n_sample:
            print('inference finished!')
            break
        processed_count += batch.shape[0]
        input = batch        
        input = input.to(device).float()
        file_name = file_name[0].split('.')[0]
        ws = get_latents(net, input, is_cars)

        os.makedirs(os.path.join(edit_directory_path, "fixed"), exist_ok=True)
        os.makedirs(os.path.join(edit_directory_path, "grid"), exist_ok=True)

        camera_lookat_point = torch.tensor(generator.rendering_kwargs['avg_camera_pivot'], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=generator.rendering_kwargs['avg_camera_radius'],
                                                  device=device)
        focal_length = 4.2647
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                                  device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        if args.edit_attribute == 'inversion':
            # img_edit = imgs
            edit_latents = ws.clone()
        else:
            img_edit, edit_latents = editor.apply_interfacegan(ws.clone(), c.repeat(ws.shape[0], 1), edit_direction,
                                                               factor=edit_degrees)

        # save latent
        torch.save(edit_latents, os.path.join(edit_directory_path, f"{file_name}_latent.pt"))

        yaw_d = math.pi / 9
        pitch_d = math.pi / 36
        yaws = [-yaw_d, yaw_d]
        pitchs = [-pitch_d, pitch_d]
 
        # print('latents mean: ', edit_latents.abs().mean())
        imgs_all = []
        for ii, pitch in enumerate(pitchs):
            for j, yaw in enumerate(yaws):
                camera_lookat_point = torch.tensor(generator.rendering_kwargs['avg_camera_pivot'], device=device)
                cam2world_pose = LookAtPoseSampler.sample(3.14 / 2 + yaw, 3.14 / 2 + pitch, camera_lookat_point,
                                                          radius=generator.rendering_kwargs['avg_camera_radius'],
                                                          device=device)
                focal_length = 4.2647
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                                          device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                imgs = generator.synthesis(edit_latents, c, noise_mode='const')['image']

                # save images
                result = tensor2im(imgs[0])
                im_save_path = os.path.join(edit_directory_path, 'fixed', f"{file_name}_y{yaw*180/math.pi:.3}_p{pitch*180/math.pi:.3}.png")
                result.save(im_save_path)

                imgs_all.append(imgs.detach().cpu())


        imgs_all = torch.cat(imgs_all, dim=0)
        # print('imgs_all:', imgs_all.shape)
        im_save_path = os.path.join(edit_directory_path, 'grid', f"{file_name}.png")
        Image.fromarray(layout_grid(imgs_all, grid_w=2, grid_h=2), 'RGB').save(im_save_path)

        if args.video:
            gen_interp_video(G=generator, mp4=f'{edit_directory_path}/{file_name}.mp4', ws=edit_latents, bitrate='10M', grid_dims=(1, 1),
                            num_keyframes=None,
                            w_frames=120, psi=0.7, truncation_cutoff=14,
                            cfg='FFHQ', image_mode='image', gen_shapes=False)



def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    opts=opts)
    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            # print('codes delta:', codes.abs().mean())
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
            # print('codes:', codes.abs().mean())
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)


def log_images(save_dir, name, im_data):
    fig = common.inference_vis_faces(im_data)
    path = os.path.join(save_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, ws, w_frames=60*4, kind='linear', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if ws.shape[0] % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = ws.shape[0] // (grid_w*grid_h)

    # all_ws = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    # for idx in range(num_keyframes*grid_h*grid_w):
    #     all_ws[idx] = ws[idx % ws.shape[0]]

    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    # zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(ws.shape[0], 1)
    # ws = G.mapping(z=zs, c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].detach().cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)

    # if gen_shapes:
    #     outdir = 'interpolation_{}_{}/'.format(all_seeds[0], all_seeds[1])
    #     os.makedirs(outdir, exist_ok=True)
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                entangle = 'camera'

                img = G.synthesis(ws=w.unsqueeze(0), c=c, noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(args.n_sample):
        camera_lookat_point = torch.tensor(g.rendering_kwargs['avg_camera_pivot'], device="cuda")
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=g.rendering_kwargs['avg_camera_radius'],
                                                  device="cuda")
        focal_length = 4.2647
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device="cuda")
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        ws = latent_codes[i].unsqueeze(0)

        gen_interp_video(G=g, mp4=f'{inversions_directory_path}/{i:05d}.mp4', bitrate='10M', grid_dims=(1, 1),
                         num_keyframes=None,
                         w_frames=300, ws=ws, psi=0.7, truncation_cutoff=14,
                         cfg='FFHQ', image_mode='image', gen_shapes=False)
        #
        # print('ws:', ws.shape)
        # print('c:', c.shape)
        imgs = g.synthesis(ws, c, noise_mode='const')['image']
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--video", action="store_true", help="save video")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    # parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
    parser.add_argument("--edit_degree", type=str, default='0', help="edit degreee")
    parser.add_argument("--edit_direction_dir", type=str, default='./editings/interfacegan_directions', help="Path to edit direction")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")
    parser.add_argument("--eg3d_generator", metavar="eg3d generator", help="path to eg3d generator path")
    parser.add_argument("--attribute_classifier", metavar="attribute classifier", help="path to attribute classifier path")

    args = parser.parse_args()
    main(args)
