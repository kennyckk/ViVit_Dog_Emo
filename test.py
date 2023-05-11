
from dataset import DogDataset
import data_transform as T
from torch.utils.data.dataloader import DataLoader
from utils import show_processed_image
train_temporal_sample = T.TemporalRandomCrop(
    16 * 16)

train_transform = T.create_video_transform(
            objective="supervised",
            input_size=224,
            is_training=False,
            scale=None,
            hflip=0.5,
            color_jitter=0.4,
            auto_augment=None,
            interpolation='bicubic',
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))


dataset=DogDataset('./data/train.csv',transform=train_transform,
                   temporal_sample=train_temporal_sample)

dataloader=DataLoader(
			dataset,
			batch_size=8,
			shuffle=True,
			)
next(iter(dataloader))
