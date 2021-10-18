import albumentations as A
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
                        A.Normalize(),
                        A.RandomCrop(height=256, width=256, always_apply=True),
                        A.OneOf(
                            [
                                A.HorizontalFlip(p=1.0),
                                A.VerticalFlip(p=1.0),
                                A.RandomRotate90(p=1.0),
                                A.Cutout()
                            ],
                            p=0.75                        
                        ),
                        ToTensorV2()
                        ], p=1.0)

val_transform = A.Compose([
                        A.Normalize(),
                        ToTensorV2()
                        ], p=1.0)

test_transform = A.Compose([
                        A.Normalize(),
                        ToTensorV2()
                        ], p=1.0)