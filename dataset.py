import json
import random
import os

import decord
import numpy as np
import torch
import pandas as pd

from einops import rearrange
from skimage.feature import hog

class_labels_map = None
cls_sample_cnt = None

def temporal_sampling(frames, start_idx, end_idx, num_samples):
	"""
	Given the start and end frame index, sample num_samples frames between
	the start and end with equal interval.
	Args:
		frames (tensor): a tensor of video frames, dimension is
			`num video frames` x `channel` x `height` x `width`.
		start_idx (int): the index of the start frame.
		end_idx (int): the index of the end frame.
		num_samples (int): number of frames to sample.
	Returns:
		frames (tersor): a tensor of temporal sampled video frames, dimension is
			`num clip frames` x `channel` x `height` x `width`.
	"""
	index = torch.linspace(start_idx, end_idx, num_samples)
	index = torch.clamp(index, 0, frames.shape[0] - 1).long()
	frames = torch.index_select(frames, 0, index)
	return frames


def numpy2tensor(x):
	return torch.from_numpy(x)


def extract_hog_features(image):
	hog_features_r = hog(image[:,:,0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
	hog_features_g = hog(image[:,:,1], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
	hog_features_b = hog(image[:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False) #visualize=True
	hog_features = np.concatenate([hog_features_r,hog_features_g,hog_features_b], axis=-1)
	hog_features = rearrange(hog_features, '(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)', ph=14, pw=14)
	return hog_features


def load_annotation_data(data_file_path):
	with open(data_file_path, 'r') as data_file:
		return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
	global class_labels_map, cls_sample_cnt
	
	if class_labels_map is not None:
		return class_labels_map, cls_sample_cnt
	else:
		cls_sample_cnt = {}
		class_labels_map = load_annotation_data(anno_pth)
		for cls in class_labels_map:
			cls_sample_cnt[cls] = 0
		return class_labels_map, cls_sample_cnt


def get_class_labels_dog(anno_pth='./dog_classmap.json'):
	global class_labels_map, cls_sample_cnt

	if class_labels_map is not None:
		return class_labels_map, cls_sample_cnt
	else:
		cls_sample_cnt = {}
		class_labels_map = load_annotation_data(anno_pth)
		for cls in class_labels_map:
			cls_sample_cnt[class_labels_map[cls]] = 0 # the 4 classes combined into 2 only
		return class_labels_map, cls_sample_cnt

def load_annotations_dog(ann_file, num_class, num_samples_per_cls): # the anotation is stored in csv format
	dataset = []
	class_to_idx, cls_sample_cnt = get_class_labels_dog()

	#print(class_to_idx, cls_sample_cnt) # check correct map and count map
	fin= pd.read_csv(ann_file, header=0)

	labels = fin['label']
	frame_dir = fin['video_id']

	for row in range(len(fin)):
		sample = {}
		sample['video'] = frame_dir[row] + ".mp4" #all frame_dir are mp4 format

		# extract the 2 labels from the 4 emotions
		class_name = labels[row]
		class_idx = class_to_idx[str(class_name)]
		class_index = int(class_idx)

		# choose a class subset of whole dataset
		if class_index < num_class:
			sample['label'] = class_index
			#if cls_sample_cnt[class_idx] < num_samples_per_cls:
			dataset.append(sample)
			cls_sample_cnt[class_idx] += 1

	return dataset


def load_annotations(ann_file, num_class, num_samples_per_cls):
	dataset = []
	class_to_idx, cls_sample_cnt = get_class_labels(num_class)
	with open(ann_file, 'r') as fin:
		for line in fin:
			line_split = line.strip().split('\t')
			sample = {}
			idx = 0
			# idx for frame_dir
			frame_dir = line_split[idx]
			sample['video'] = frame_dir
			idx += 1
								
			# idx for label[s]
			label = [x for x in line_split[idx:]]
			assert label, f'missing label in line: {line}'
			assert len(label) == 1
			class_name = label[0]
			class_index = int(class_to_idx[class_name])
			
			# choose a class subset of whole dataset
			if class_index < num_class:
				sample['label'] = class_index
				if cls_sample_cnt[class_name] < num_samples_per_cls:
					dataset.append(sample)
					cls_sample_cnt[class_name]+=1

	return dataset


class DecordInit(object):
	"""Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

	def __init__(self, num_threads=1, **kwargs):
		self.num_threads = num_threads
		self.ctx = decord.cpu(0)
		self.kwargs = kwargs
		
	def __call__(self, filename):
		"""Perform the Decord initialization.
		Args:
			results (dict): The resulting dict to be modified and passed
				to the next transform in pipeline.
		"""
		reader = decord.VideoReader(filename,
									ctx=self.ctx,
									num_threads=self.num_threads)
		return reader

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'sr={self.sr},'
					f'num_threads={self.num_threads})')
		return repr_str

def skip_bad_collate (batch):
	batch=list(filter(lambda x: x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)

class DogDataset(torch.utils.data.Dataset):
	"""Load the Dog Video Files"""
	def __init__(self,
				 annotation_path,
				 num_frames=16,
				 num_samples_per_cls=60,
				 num_class=2,
				 transform=None,
				 temporal_sample=None,
				 all_frames=None):
		#self.configs = configs
		self.num_class=num_class
		self.num_frames=num_frames
		self.num_samples_per_cls=num_samples_per_cls
		self.annotation_path = annotation_path
		self.data = load_annotations_dog(self.annotation_path, self.num_class, self.num_samples_per_cls)
		self.all_frames=all_frames

		self.transform = transform
		self.temporal_sample = temporal_sample
		self.target_video_len = self.num_frames

		self.v_decoder = DecordInit()
	def __getitem__(self, index):
		curr_path=os.path.dirname(self.annotation_path)
		while True:
			try:
				path = self.data[index]['video']
				path=os.path.join(curr_path,'videos', path) #correct the path to data folder
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)

				if self.all_frames is None:
					# Sampling video frames
					start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
					assert end_frame_ind - start_frame_ind >= self.target_video_len
					frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
					video = v_reader.get_batch(frame_indice).asnumpy()
					
				else:
					#this is to get all frames from the video up to max length defined
					if total_frames>self.all_frames:#just get the max length that can be taken
						frame_indice=np.linspace(0,self.all_frames-1,self.all_frames,dtype=int)
						video=v_reader.get_batch(frame_indice).asnumpy()
						#print('the video is over specified frames', video.shape)
						
					else: # will need to pad the rest of the frames
						frame_indice=np.linspace(0,total_frames-1,total_frames,dtype=int)
						video=v_reader.get_batch(frame_indice).asnumpy() #in numpy array T,H,W,C
						v_shape=video.shape
						#print('the video is under specified frames', v_shape)
						pad_shape=(self.all_frames-total_frames,v_shape[1],v_shape[2],v_shape[3]) #prepare a shape for the pads
						pads=np.zeros(pad_shape)

						video=np.concatenate((video,pads),axis=0)
						#print('the video is under specified frames', video.shape)
				del v_reader
				break		
			except Exception as e:
				print(e, 'skipping this data instance')
				#index = random.randint(0, len(self.data) - 1)
				return None

		# Video align transform: T C H W
		with torch.no_grad():
			video = torch.from_numpy(video).permute(0, 3, 1, 2)
			if self.transform is not None:
				video = self.transform(video)

		label = self.data[index]['label'] # only supervise objective is doing

		return video, label
	def __len__(self):
		return len(self.data)


#just a unit test to show the image of the transformed image
if __name__=="__main__":
	import data_transform as T
	from utils import show_processed_image

	train_temporal_sample = T.TemporalRandomCrop(
    16 * 16,full_length=True)

	transform= T.transforms_train_dog(img_size=224,
                    augmentation=True,
					crop_pct=None,
					 hflip=1, # 0 for non-augment data
					 auto_augment=None,
					 interpolation='bilinear',
					 rotate=1,
					 noise=1
					 )
	# transform=T.transforms_eval(img_size=224,
	# 				crop_pct=None,
	# 				interpolation='bicubic',
	# 				mean=(0.45, 0.45, 0.45),
	# 				std= (0.225, 0.225, 0.225))


	sample_videos=DogDataset('./face_data/eval.csv',
				 num_frames=16,
				 num_samples_per_cls=60,
				 num_class=2,
				 transform=transform,
				 temporal_sample=train_temporal_sample,
				 all_frames=256)

	sample=next(iter(sample_videos))[0]
	#print(sample)
	show_processed_image(sample.permute(0,2,3,1),'./dummies/',mean=(0.45, 0.45, 0.45),std=(0.225, 0.225, 0.225))