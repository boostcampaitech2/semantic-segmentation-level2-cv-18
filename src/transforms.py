import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 512


# val, test
tf = A.Compose([
		A.Resize(IMAGE_SIZE, IMAGE_SIZE),
		A.Normalize(),
		ToTensorV2()
])


# train
train_tf = A.Compose([
		A.Resize(IMAGE_SIZE, IMAGE_SIZE),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.OneOf([
			A.ElasticTransform(p=1, alpha=40, sigma=120 * 0.05, alpha_affine=120 * 0.03),
			A.GridDistortion(p=1),
			A.CLAHE(p=1),
		], p=0.6),
		A.RandomBrightness(p=0.5),
		A.Rotate(limit=30, p=0.5),
		A.Normalize(),
		ToTensorV2()							
])
