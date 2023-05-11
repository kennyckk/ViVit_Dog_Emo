import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from dataset import Kinetics,DogDataset
import data_transform as T

class Collator(object):

	def __init__(self, objective):
		self.objective = objective
	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		mask_list = []
		marker_list = []
		for record in minibatch:
			image_list.append(record[0])
			label_list.append(record[1])
			if self.objective == 'mim':
				mask_list.append(record[2])
				marker_list.append(record[3])
		minibatch = []
		minibatch.append(torch.stack(image_list))
		if self.objective == 'mim':
			minibatch.append(torch.stack(label_list))
			minibatch.append(torch.stack(mask_list))
			minibatch.append(marker_list)
		else:
			label = np.stack(label_list)
			minibatch.append(torch.from_numpy(label))
		
		return minibatch

class DogDataModule(pl.LightningDataModule):
	def __init__(self,
				 batch_size,
				 train_ann_path,
				 val_ann_path=None,
				 test_ann_path=None,
				 num_workers=0,
				 data_statics='kinetics',
				 objective='supervised',
				 img_size=224,
				 auto_augment=None,
				 num_frames=16,
				 frame_interval=16
				 ):
		super().__init__()
		self.train_ann_path = train_ann_path
		self.val_ann_path = val_ann_path
		self.test_ann_path = test_ann_path
		self.data_statics=data_statics
		self.objective=objective
		self.img_size=img_size
		self.auto_augment=auto_augment
		self.num_frames=num_frames
		self.frame_interval=frame_interval
		self.batch_size=batch_size
		self.num_workers=num_workers

	# function to get every data instance of the video from the annotation csv file
	def get_dataset(self, annotation_path, transform, temporal_sample):
		dataset = DogDataset(
			annotation_path,
			transform=transform,
			temporal_sample=temporal_sample)

		return dataset

	def setup(self, stage):
		color_jitter = 0.4
		scale = None

		if self.data_statics == 'imagenet':
			mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		elif self.data_statics == 'kinetics':
			mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
		else:
			mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

		# prepare the transformation needed for training data
		train_transform = T.create_video_transform(
			objective=self.objective,
			input_size=self.img_size,
			is_training=True,
			scale=scale,
			hflip=0.5,
			color_jitter=color_jitter,
			auto_augment=self.auto_augment,
			interpolation='bicubic',
			mean=mean,
			std=std)

		#prepare sampling of frames from videos
		train_temporal_sample = T.TemporalRandomCrop(
		self.num_frames * self.frame_interval)

		# call out the DogDataset to process the training data
		self.train_dataset = self.get_dataset(
			self.train_ann_path,
			train_transform,
			train_temporal_sample)

		# to prepare data augmentation



		#prepare the transformation needed for eval data
		assert (self.val_ann_path is not None), "please input valid ann eval path"

		val_transform = T.create_video_transform(
			input_size=self.img_size,
			is_training=False,
			interpolation='bicubic',
			mean=mean,
			std=std)
		# prepare sampling of frames from videos
		val_temporal_sample = T.TemporalRandomCrop(
			self.num_frames * self.frame_interval)
		# call out the DogDataset to process the eval data
		self.val_dataset = self.get_dataset(
			self.val_ann_path,
			val_transform,
			val_temporal_sample)

		# load in training dataloader
	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=True,
			pin_memory=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=False,
			)



class KineticsDataModule(pl.LightningDataModule):
	def __init__(self, 
				 configs,
				 train_ann_path,
				 val_ann_path=None,
				 test_ann_path=None,
				 ):
		super().__init__()
		self.train_ann_path = train_ann_path
		self.val_ann_path = val_ann_path
		self.test_ann_path = test_ann_path
		self.configs = configs

	def get_dataset(self, annotation_path, transform, temporal_sample):
		dataset = Kinetics(
			self.configs,
			annotation_path,
			transform=transform,
			temporal_sample=temporal_sample)
		
		return dataset

	def setup(self, stage):
		if self.configs.objective == 'mim':
			scale = (0.5, 1.0)
			color_jitter = None
		else:
			color_jitter = 0.4
			scale = None
		
		if self.configs.data_statics == 'imagenet':
			mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		elif self.configs.data_statics == 'kinetics':
			mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
		else:
			mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
		
		train_transform = T.create_video_transform(
			objective=self.configs.objective,
			input_size=self.configs.img_size,
			is_training=True,
			scale=scale,
			hflip=0.5,
			color_jitter=color_jitter,
			auto_augment=self.configs.auto_augment,
			interpolation='bicubic',
			mean=mean,
			std=std)
		train_temporal_sample = T.TemporalRandomCrop(
			self.configs.num_frames * self.configs.frame_interval)
			
		self.train_dataset = self.get_dataset(
			self.train_ann_path,
			train_transform,
			train_temporal_sample)
		
		if self.val_ann_path is not None:
			val_transform = T.create_video_transform(
				input_size=self.configs.img_size,
				is_training=False,
				interpolation='bicubic',
				mean=mean,
				std=std)
			val_temporal_sample = T.TemporalRandomCrop(
				self.configs.num_frames * self.configs.frame_interval)
			self.val_dataset = self.get_dataset(
				self.val_ann_path,
				val_transform,
				val_temporal_sample)
		
		if self.test_ann_path is not None:
			# need to update
			test_transform = T.Compose([
				T.Resize(scale_range=(-1, 256)),
				T.ThreeCrop(size=self.configs.img_size),
				T.ToTensor(),
				T.Normalize(mean, std),
				])
			test_temporal_sample = T.TemporalRandomCrop(
				self.configs.num_frames * self.configs.frame_interval)
			self.test_dataset = self.get_dataset(
				self.test_ann_path,
				test_transform,
				test_temporal_sample)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			collate_fn=Collator(self.configs.objective).collate,
			shuffle=True,
			drop_last=True, 
			pin_memory=True
		)
	
	def val_dataloader(self):
		if self.val_ann_path is not None:
			return DataLoader(
				self.val_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				collate_fn=Collator(self.configs.objective).collate,
				shuffle=False,
				drop_last=False,
			)
	
	def test_dataloader(self):
		if self.test_ann_path is not None:
			return DataLoader(
				self.test_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				collate_fn=Collator(self.configs.objective).collate,
				shuffle=False,
				drop_last=False,
			)