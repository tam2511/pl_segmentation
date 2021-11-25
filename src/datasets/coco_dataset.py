from typing import Optional, Callable
import os
import sys
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from datasets.utils import read_image


class CocoDataset(Dataset):
    def __init__(self, images_path: str, annos_path: str, transform: Optional[Callable] = None):
        self.annos_path = annos_path
        self.images_path = images_path
        self.transform = transform
        sys.stdout = open(os.devnull, 'w')
        self.coco = COCO(annos_path)
        sys.stdout = sys.__stdout__
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = read_image(os.path.join(self.images_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = self._load_image(id)
        annos = self._load_target(id)
        mask = np.zeros(image.shape)
        for anno in annos:
            mask += self.coco.annToMask(anno)
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image, mask = result['image'], result['mask']
        return image, mask

    def __len__(self) -> int:
        return len(self.ids)
