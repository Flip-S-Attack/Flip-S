import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
import numpy as np
from PIL import Image
import os.path
import pickle
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class my_ImageDataset(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        processor: Optional[Callable] = None,
        use_ir = False,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.processor = processor
        
        self.use_ir = use_ir
        self.normalize = transforms.Normalize(mean=self.processor.image_mean,std=self.processor.image_std)
        
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.processor is not None:
            sample = self.processor(images=sample, return_tensors="pt", do_normalize=False)['pixel_values'].squeeze(0)
            if not self.use_ir:
                sample = self.normalize(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def load_split_ImageNet1k_valid(valdir, aux_num=512, seed=0, processor=None, batch_size=64, shuffle=False, num_workers=1, pin_memory=False,split_ratio=0.,use_ir=False,use_defalut_valid_batchsize=False):
    val_dataset = my_ImageDataset(valdir, processor=processor, use_ir=use_ir)
    total = len(val_dataset)
    val_dataset, aux_dataset = random_split(
        dataset=val_dataset,
        lengths=[total-aux_num, aux_num],
        generator=torch.Generator().manual_seed(seed)
    )
    valid_batch_size = 128 if use_defalut_valid_batchsize else batch_size
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=valid_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    aux_loader = torch.utils.data.DataLoader(
            aux_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    if split_ratio == 0.:
        return val_loader, aux_loader
    elif split_ratio > 0. and split_ratio <= 1.:
        total = len(val_dataset)
        small_val_dataset, _ = random_split(
            dataset=val_dataset,
            lengths=[int(total*split_ratio), total-int(total*split_ratio)],
            generator=torch.Generator().manual_seed(seed)
        )
        small_val_loader = torch.utils.data.DataLoader(
            small_val_dataset,
            batch_size=valid_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return val_loader, aux_loader, small_val_loader
    else:
        raise ValueError