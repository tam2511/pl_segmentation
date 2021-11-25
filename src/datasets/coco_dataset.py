from typing import Optional, Callable
import os
import contextlib
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
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.coco = COCO(annos_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_classes = 1 + len(self.coco.getCatIds())

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
        mask = np.zeros((self.num_classes, image.shape[0], image.shape[1]), dtype=np.uint8)
        for anno in annos:
            class_id = anno['category_id']
            mask[class_id, :, :] = self.__annotation2binarymask(mask[class_id], anno)
        if self.transform:
            mask = mask.transpose(1, 2, 0)
            result = self.transform(image=image, mask=mask)
            image, mask = result['image'], result['mask']
            mask = mask.permute(2, 0, 1)
        mask[0, :, :] = (mask[1:, :, :].sum(0) == 0)
        return image, mask

    def __len__(self) -> int:
        return len(self.ids)

    def __decodeSeg(self, mask, segmentations):
        """
        Draw segmentation
        """
        pts = [
            np
                .array(anno)
                .reshape(-1, 2)
                .round()
                .astype(int)
            for anno in segmentations
        ]
        mask = cv2.fillPoly(mask, pts, 1)

        return mask

    def __decodeRl(self, mask, rle):
        """
        Run-length encoded object decode
        """
        mask = mask.reshape(-1, order='F')

        last = 0
        val = True
        for count in rle['counts']:
            val = not val
            mask[last:(last + count)] |= val
            last += count

        mask = mask.reshape(rle['size'], order='F')
        return mask

    def __annotation2binarymask(self, mask, annotation):
        segmentations = annotation['segmentation']
        if isinstance(segmentations, list):  # segmentation
            return self.__decodeSeg(mask, segmentations)
        else:  # run-length
            return self.__decodeRl(mask, segmentations)
