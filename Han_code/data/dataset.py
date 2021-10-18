from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os, cv2
import numpy as np

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, dataset_path='../input', mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = dataset_path
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        # transform -> albumentations 라이브러리 활용
        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]
            masks = transformed["mask"]
            
        return images, masks, image_infos
        

    def __len__(self) -> int:
        return len(self.coco.getImgIds())