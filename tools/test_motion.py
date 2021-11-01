import os
import shutil
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.ops import box_convert
import itertoolsT
from collections import defaultdict

from yolox.data import (
    MOTDataset,
    MotionDataset,
    TrainTransform,
    YoloBatchSampler,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
)
from yolox.data import DataPrefetcher




dataset_folder = '/workspace/ByteTrack/datasets/FMV/motion'
anno_file = 'anno.json'
split_file = 'splits.json'
is_distributed = True
num_gpus = 8
batch_size = 64
seed = None
input_size = (736, 1280)
debug_limit = 100
data_num_workers = 8

start_epoch = 0
no_aug_epochs = 1
max_epoch = 2

save_image_examples = True
max_labels = 300
distort = True
scale_bbox_width = 0.95
scale_bbox_height = 0.95
preproc=TrainTransform(
    rgb_means=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_labels=max_labels,
    distort=distort,
    scale_bbox_width=scale_bbox_width,
    scale_bbox_height=scale_bbox_height,
)
#preproc=None
mosaic_dataset = True
no_aug = False
degrees = 10.0
translate = 0.1
scale = (0.1, 2)
shear = 2.0
perspective = False
enable_mixup = False


dataset = MotionDataset(
    root_path=dataset_folder,
    anno_file=anno_file,
    split_file=split_file,
    is_train=True,
    img_size=input_size,
    preproc=preproc,
    debug_limit=debug_limit,
    save_image_examples=save_image_examples,
)

print('Finished creating MotionDataset dataset.')

if mosaic_dataset:
    dataset = MosaicDetection(
        dataset,
        mosaic=not no_aug,
        img_size=input_size,
        preproc=preproc,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        enable_mixup=enable_mixup,
        save_image_examples=save_image_examples,
    )

    print('Finished creating MosaicDetection dataset')

if is_distributed:
    #batch_size = batch_size // dist.get_world_size()
    batch_size = batch_size // num_gpus

# TODO: improve sampler
sampler = InfiniteSampler(
    len(dataset), seed=seed if seed else 0
)

batch_sampler = YoloBatchSampler(
    sampler=sampler,
    batch_size=batch_size,
    drop_last=False,
    input_dimension=input_size,
    mosaic=not no_aug,
)

print('Finished creating batched_sampler.')

dataloader_kwargs = {"num_workers": data_num_workers, "pin_memory": True}
dataloader_kwargs["batch_sampler"] = batch_sampler
#dataloader_kwargs["collate_fn"] = collator
train_loader = DataLoader(dataset, **dataloader_kwargs)
print(f'Finished creating train_loader. len: {len(train_loader)}')

#train_loader.close_mosaic()

assert torch.cuda.is_available()
device = torch.cuda.device(0)
prefetcher = DataPrefetcher(train_loader)
print('Finished creating prefetcher.')
start_iter = 1
max_iter = len(train_loader)




data_type = torch.float16
for epoch in range(start_epoch, max_epoch):
    print(f'epoch: {epoch}')
    for iteration in range(max_iter): # why 73?
        print(f'\titeration: {iteration}')
        inps, targets = prefetcher.next()
        inps = inps.to(data_type)        
        targets = targets.to(data_type) 



print('End.')