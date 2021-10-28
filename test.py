import os
import random
import json
import yaml
import argparse
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from tqdm import tqdm

# 전처리를 위한 라이브러리
import torchvision
import albumentations as A

# 시각화를 위한 라이브러리
import wandb

from importlib import import_module
from pprint import pprint

# For TTA
import ttach as tta

# IMPORT CUSTOM MODULES
from util.ploting import plot_examples, plot_train_dist
from data.dataloader import get_dataloaders
from train import train

# GPU 사용 가능 여부에 따라 device 정보 저장
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_ver_n_settings():
    print("-"*30)
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    print("-"*30)
    

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg-yaml", type=str, default="./test.yaml", help="Choose your own cfg yaml.")
    
    args = parser.parse_args()
    
    return args


def cfg_check(cfg):
    selected_framework = cfg["SELECTED"]["FRAMEWORK"]
    assert(selected_framework in cfg["FRAMEWORKS_AVAILABLE"])
    assert(cfg["SELECTED"]["MODEL"] in cfg["MODELS_AVAILABLE"][selected_framework])    
    
    
def get_cfg_from(args):
    with open(args.cfg_yaml, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if cfg["EXPERIMENTS"]["LEARNING_RATE"]:
        cfg["EXPERIMENTS"]["LEARNING_RATE"] = float(cfg["EXPERIMENTS"]["LEARNING_RATE"])
    if cfg["SELECTED"]["OPTIMIZER_CFG"]["weight_decay"]:
        cfg["SELECTED"]["OPTIMIZER_CFG"]["weight_decay"] = float(cfg["SELECTED"]["OPTIMIZER_CFG"]["weight_decay"])
    
    if not os.path.exists(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"]):
        os.mkdir(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"])
    if not os.path.exists(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"]):
        os.mkdir(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"])
    
    cfg_check(cfg)
    
    return cfg


def get_df_train_categories_counts(cfg):
    dataset_path = cfg["DATASET"]["PATH"]
    train_file_path = os.path.join(dataset_path, cfg["DATASET"]["TRAIN_FILE_NAME"])

    # Read annotations
    with open(train_file_path, 'r') as f:
        dataset = json.loads(f.read())

    df = pd.DataFrame(dataset["annotations"])
    df = df[["id", "category_id"]]
    df = df.groupby(by="category_id", as_index=False).count()
    df["category_id"] = df["category_id"].apply(lambda x: dataset["categories"][x-1]["name"])
    df.columns = ["Categories", "Number of annotations"]
    df = df.sort_values(by="Number of annotations", ascending=False)
    
    return df


def add_bg_index_to(df):
    # df 에 index(Background) 를 추가한다. 
    df_target = pd.DataFrame(["Background"], columns=["Categories"])
    df_target = df_target.append(df)
    df_target = df_target.sort_index()
    df_target.index = range(df_target.shape[0])
    return df_target


# 제공받은 baseline
def fix_seed_as(random_seed:int=21):
    """Seed 를 고정해서 일관된 실험 결과를 얻는다."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# 제공받은 baseline
def eda(cfg):
    dataset_path = cfg["DATASET"]["PATH"]
    anns_file_path = os.path.join(dataset_path, cfg["DATASET"]["ANNS_FILE_NAME"])

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']

    df_categories = pd.DataFrame(categories)

    num_categories = len(df_categories["name"])
    num_super_categories = len(df_categories["supercategory"])
    num_annotations = len(anns)
    num_images = len(imgs)

    print("-"*30)
    print('Number of super categories:', num_super_categories)
    print('Number of categories:', num_categories)
    print('Number of annotations:', num_annotations)
    print('Number of images:', num_images)
    print("-"*30)
    
    
def wnb_init(cfg):
    exp_cfg = cfg["EXPERIMENTS"]
    if exp_cfg["WNB_TURN_ON"]:
        return wandb.init(**exp_cfg["WNB_INIT"])
    

def set_torchvision_model(cfg, model):
    num_classes = cfg["DATASET"]["NUM_CLASSES"]
    model_selected = cfg["SELECTED"]["MODEL"]
    
    if model_selected == "lraspp_mobilenet_v3_large":
        model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "deeplab" in model_selected:
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif "fcn" in model_selected:
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model


###################################
## set_torchvision_model 함수처럼 짤 수 있으면 좋을 것 같습니다.
def set_smp_model(cfg, model):
    return model
###################################

def is_same_width(t1, t2):
    return t1.shape[-2] == t2.size(-2)

def is_same_height(t1, t2):
    return t1.shape[-1] == t2.size(-1)

def simple_check(cfg, model):
    # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
    x = torch.randn([2, 3, 512, 512])
    out = model(x)['out']
    assert(is_same_width(x, out) and is_same_height(x, out))
    assert(out.size(-3) == cfg["DATASET"]["NUM_CLASSES"])
    
def get_trainable_model(cfg):
    cfg_selected = cfg["SELECTED"]
    model_selected = cfg_selected["MODEL"]
    frame_selected = cfg_selected["FRAMEWORK"]
    num_classes = cfg["DATASET"]["NUM_CLASSES"]
    
    if frame_selected == "torchvision":
        model = getattr(torchvision.models.segmentation, model_selected)(**cfg_selected["MODEL_CFG"])
        model = set_torchvision_model(cfg, model)
    elif cfg_selected["FRAMEWORK"] == "segmentation_models.pytorch":
        model = getattr(smp, model_selected)(**cfg_selected["MODEL_CFG"])
        model = set_smp_model(cfg, model)
    
    simple_check(cfg, model)
    
    return model


def get_trained_model(cfg, device):
    model = get_trainable_model(cfg)
    # best model 저장된 경로
    model_path = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
                              f"{cfg['SELECTED']['MODEL']}.pt")

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    return model
    

# 제공받은 baseline -> wandb 등 custom
def test(model, test_dataloader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_dataloader)):
            
            # inference (512 x 512)
            # outs = model(torch.stack(imgs).to(device))
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    
    if cfg["EXPERIMENTS"]["WNB_TURN_ON"]:
        plot_examples(model=model,
                      cfg=cfg,
                      device=DEVICE,
                      mode="val", 
                      batch_id=0, 
                      num_examples=8, 
                      dataloaer=val_dataloader)
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array    
    

def create_submission(model, test_dataloader, device, cfg):
    # sample_submisson.csv 열기
    sample_submission_file = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"],
                                          "sample_submission.csv")
    submission = pd.read_csv(sample_submission_file, index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_dataloader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, 
                                        "PredictionString" : ' '.join(str(e) for e in string.tolist())},
                                        ignore_index=True)

    # submission.csv로 저장
    submission_file = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"], 
                                   f"{cfg['SELECTED']['MODEL']}.csv")
    submission.to_csv(submission_file, index=False)
    
    
if __name__ == "__main__":
    print_ver_n_settings()
    cfg = get_cfg_from(get_args())
    pprint(cfg)
    fix_seed_as(cfg["SEED"])
    
    eda(cfg)
    
    wnb_run = wnb_init(cfg)
    
    df_train_categories_counts = get_df_train_categories_counts(cfg)
    plot_train_dist(cfg, df_train_categories_counts)
    sorted_df_train_categories_counts = add_bg_index_to(df_train_categories_counts)
    category_names = sorted_df_train_categories_counts["Categories"].to_list()
    
    model = get_trainable_model(cfg)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg, category_names)
    
    train(num_epochs=cfg["EXPERIMENTS"]["NUM_EPOCHS"], 
          model=model, 
          train_dataloader=train_dataloader, 
          val_dataloader=val_dataloader, 
          criterion=nn.CrossEntropyLoss(), 
          optimizer=torch.optim.Adam(params = model.parameters(), 
                                     lr=cfg["EXPERIMENTS"]["LEARNING_RATE"], 
                                     weight_decay=1e-6), 
          saved_dir=cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
          val_every=cfg["EXPERIMENTS"]["VAL_EVERY"], 
          device=DEVICE,
          category_names=category_names,
          cfg=cfg)
    
    model_trained = get_trained_model(cfg, DEVICE)
    if cfg["EXPERIMENTS"]["TTA_TURN_ON"]:
        tta_cfg = cfg["EXPERIMENTS"]["TTA_CFG"]
        tta_trans = tta.Compose([getattr(tta.transforms, trans)(**cfg) for trans, cfg in tta_cfg.items()])
        model_trained = tta.SegmentationTTAWrapper(model=model_trained, 
                                                   transforms=tta_trans,
                                                   merge_mode="mean")
        model_trained.to(DEVICE)
    
    create_submission(model=model_trained, 
                      test_dataloader=test_dataloader, 
                      device=DEVICE,
                      cfg=cfg)
    
    if wnb_run is not None:
        wnb_run.finish()
    