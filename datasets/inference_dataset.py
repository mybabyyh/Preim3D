import os.path

import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import json


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None, preprocess=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		_, filename = os.path.split(from_path)

		return from_im, filename


class InferenceCameraDataset(Dataset):

	def __init__(self, source_root, camera_data_json, opts, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.source_transform = source_transform
		self.opts = opts
		self.camera_dict = {}
		with open(os.path.abspath(camera_data_json), 'r', encoding='utf8') as fp:
			camera_param = json.load(fp)
			for img in camera_param['labels']:
				self.camera_dict[img[0]] = img[1]

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		if self.source_transform:
			from_im = self.source_transform(from_im)

		_, filename = os.path.split(from_path)
		if filename not in self.camera_dict:
			c = torch.zeros(25)
			return from_im, c, ''
		c = torch.tensor(self.camera_dict[filename])

		return from_im, c, filename


class InferenceFFHQEG3DDataset(Dataset):

	def __init__(self, source_root, camera_data_json, opts, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.source_transform = source_transform
		self.opts = opts
		self.camera_list = []
		with open(os.path.abspath(camera_data_json), 'r', encoding='utf8') as fp:
			camera_param = json.load(fp)
			for img in camera_param['labels']:
				self.camera_list.append(img[1])

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		if self.source_transform:
			from_im = self.source_transform(from_im)

		_, filename = os.path.split(from_path)
		idx = int(filename.split('.')[0][3:])
		c = torch.tensor(self.camera_list[idx])

		return from_im, c, filename

