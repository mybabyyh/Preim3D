import os.path

import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import json


class ImagesDatasetCars(Dataset):

	def __init__(self, source_root, target_root, camera_data_json, pose_data_json, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self.camera_dict = {}
		with open(os.path.abspath(camera_data_json), 'r', encoding='utf8') as fp:
			camera_param = json.load(fp)
			for img in camera_param['labels']:
				key = img[0].split('/')[1]
				self.camera_dict[key] = img[1]

		print('self.camera_dict:', len(self.camera_dict))
		self.pose_dict = {}
		# if pose_data_json is not None:
		# 	with open(os.path.abspath(pose_data_json), 'r', encoding='utf8') as fp:
		# 		pose_param = json.load(fp)
		# 		for key, value in pose_param.items():
		# 			self.pose_dict[key] = value['angle'][0]
		# 			# print(f'{key}: {value["angle"][0]}')

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		if self.opts.dataset_type == 'cars_encode':
			path1 = os.path.abspath(from_path).split('/')[-3]
			path2 = os.path.abspath(from_path).split('/')[-2]
			path3 = os.path.abspath(from_path).split('/')[-1]
			filename = '/'.join([path1, path2, path3])
		else:
			_, filename = os.path.split(from_path)
		# print('filename:', filename)
		if filename not in self.camera_dict:
			print(f'{filename} not camera param')
			c = torch.zeros(25)
			pose = torch.tensor([999., 999., 999.])
			return from_im, to_im, c, pose
		c = torch.tensor(self.camera_dict[filename])
		# print('c: ', c.shape)
		# print(f'{filename}: {self.camera_dict[filename][:4]}')
		# print(f'{filename}: {self.pose_dict[filename]}')
		if filename not in self.pose_dict:
			# print('filename: ', filename)
			# print('self.pose_dict: ', self.pose_dict[filename])
			pose = torch.tensor([999., 999., 999.])
		else:
			pose = torch.tensor(self.pose_dict[filename])

		return from_im, to_im, c, pose
