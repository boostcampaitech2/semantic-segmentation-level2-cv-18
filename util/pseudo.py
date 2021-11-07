import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools import mask
from skimage import measure

def mask2coco(cfg):
    # csv load
    df = pd.read_csv(cfg['csv_file_path'])
    # test_json load
    with open(cfg['test_file_path']) as f:
        data = json.load(f)
    del data['annotations']
    data['annotations']=[]
    # image_id 
    image_id = 0
    seg_id = 0
    # class 11 
    categoryNum = 11
    for item in tqdm(df.itertuples()):
        # 이미지 이름
        name = item[1]
        # csv 원본 mask (str)
        segmask = item[2]
        # list 변환
        maskOrigin = np.array(segmask.split()).reshape(cfg['maxWidth'],cfg['maxHeight'])
        # 각각의 class 찾아서 저장해놓음 [category][height][width]
        masklist = [[[0 for _ in range(cfg['maxWidth'])] for _ in range(cfg['maxHeight'])] for _ in range(categoryNum)]
        existCategory = set() # 카테고리 존제 유무
        for height in range(cfg['maxHeight']):
            for width in range(cfg['maxWidth']):
                if(maskOrigin[height][width] == '0'):
                    continue
                category= int(maskOrigin[height][width])
                masklist[category][height][width] = '1'
                existCategory.add(category)
        existCategory = sorted(list(existCategory))
        # 각 카테고리 별로 데이터 삽입
        for category in existCategory:
            # annotation format
            annotation = {
                    "segmentation": [],
                    "area": 0,
                    "iscrowd": 0,
                    "image_id": 0,
                    "bbox": 0,
                    "category_id": 0,
                    "id": 0
                }
            #
            ground_truth_binary_mask = np.array(masklist[category],dtype=np.uint8)
            fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(ground_truth_binary_mask,0.5)
            annotation["area"] = ground_truth_area.tolist()
            annotation["bbox"] = ground_truth_bounding_box.tolist()
            annotation["image_id"] = image_id
            annotation["id"] = seg_id
            annotation["category_id"] = category
            seg_id+=1
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)
            data['annotations'].append(annotation)
        image_id+=1

    with open(cfg['result_file_path'],'w') as fp:
        json.dump(data,fp,indent=4)
        
# segmentation length == 4 일경우 delete 해줘야함
def cocofix(cfg):
    #Open JSON
    val_json = open(cfg['result_file_path'], "r+")
    json_object = json.load(val_json)
    val_json.close()

    for i, instance in enumerate(json_object["annotations"]):
        if len(instance["segmentation"][0]) == 4:
            print("instance number", i, "raises arror:", instance["segmentation"][0]) 
            del instance["segmentation"][0]

def main():
    # config
    cfg = {
        "csv_file_path" : "../segmentation/submission/sample_submission.csv",
        "test_file_path" : "../segmentation/input/data/test.json",
        "result_file_path" : "../segmentation/baseline_code/pseudo.json",
        "maxWidth" : 256,
        "maxHeight" : 256,
    }
    mask2coco(cfg)
    cocofix(cfg)
    
if __name__ == "__main__":
    main()


