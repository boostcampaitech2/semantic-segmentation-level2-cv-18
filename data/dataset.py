import os

import numpy as np

import cv2
from pycocotools.coco import COCO

from torch.utils.data import Dataset



def get_classname(classID, categories):
    for categoty in categories:
        if categoty['id'] == classID:
            return categoty['name']
    return "None"
    

class CustomDataSet(Dataset):
    """COCO format"""
    def __init__(self, data_dir, category_names, cfg, mode='train', transform=None):
        super(CustomDataSet, self).__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.category_names = category_names
        self.dataset_path = cfg["DATASET"]["PATH"]
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if self.mode in ('train', 'val'):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            category_ids = self.coco.getCatIds()
            categories = self.coco.loadCats(category_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            for ann in anns:
                class_name = get_classname(ann['category_id'], categories)
                pixel_value = self.category_names.index(class_name)
                masks[self.coco.annToMask(ann) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())