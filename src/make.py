import torch
import json, os, cv2, copy
import pandas as pd
import numpy as np
from tqdm import tqdm


# make augmix.npy for augmix
def make_augmix(train_loader):
	print('Start making augmix.npy...')
	class_dict = {3: [], 4: [], 5: [], 9: [], 10: []}
	tmp_image = []
	tmp_mask = []

	for step, (tmp_images, tmp_masks, _) in enumerate(tqdm(train_loader)):
		for i in range(len(tmp_masks)):
			mask = tmp_masks[i].numpy().astype(np.uint8)
			img = tmp_images[i].permute([1, 2, 0]).numpy().astype(np.float64)
			mask[mask == 1] = 0
			mask[mask == 2] = 0
			mask[mask == 6] = 0
			mask[mask == 7] = 0
			mask[mask == 8] = 0
			class_type = np.unique(mask)
			if len(class_type) == 1:
				continue
			mask3d = np.dstack([mask]*3)
			res = cv2.bitwise_and(img, img, mask=mask)
			for j in class_type:
				if j == 0:
					continue
				tmp_mask = copy.deepcopy(mask)
				tmp_mask[tmp_mask != j] = 0
				tmp = copy.deepcopy(res)
				tmp = cv2.bitwise_and(tmp, tmp, mask=tmp_mask)
				class_dict[j].append(tmp)
	np.save('augmix.npy', class_dict)
	print('Finish making augmix.npy!')


# make submission csv file
def make_submission(submission_dir, save_name, file_name_list, preds_array):
	print('\nStart making submission...')
	if not os.path.exists(submission_dir):
		os.makedirs(submission_dir)
	file_names = [y for x in file_name_list for y in x]
	submission = pd.read_csv('/opt/ml/submission/sample_submission.csv', index_col=None)
	
	for file_name, string in zip(file_names, preds_array):
		submission = submission.append(
				{"image_id" : file_name,
				"PredictionString" : ' '.join(str(e) for e in string.tolist())},
				ignore_index=True
				)
	submission.to_csv(
			os.path.join(submission_dir, save_name + '.csv'), index=False
			)
	print('Finish making submission!')
