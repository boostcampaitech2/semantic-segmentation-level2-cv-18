import os
from tqdm import tqdm
from pprint import pprint
import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import albumentations as A

import torch

from data.dataloader import (
    get_dataloaders
)
from config.read_config import (
    print_ver_n_settings, get_args, get_cfg_from
)
from config.wnb import(
    wnb_init
)
from util.eda import (
    get_df_train_categories_counts, add_bg_index_to
)
from util.ploting import (
    plot_examples
)
from train import (
    get_trainable_model
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
def inference(model, test_dataloader, device, cfg):
    """By using trained model, infer all images in test_dataloader
    Args:
        model: trained model
        test_dataloader : DataLoader
        device: inference place
        
    Returns:
        file_names: the file names for all images in test_dataloader
        preds_array: the model prediction for all images in test_dataloader
    """
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
                      device=device,
                      mode="test", 
                      batch_id=0, 
                      num_examples=8, 
                      dataloaer=test_dataloader)
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array    
    

# 제공받은 baseline -> cfg 적용
def create_submission(model, test_dataloader, device, cfg):
    # sample_submisson.csv 열기
    sample_submission_file = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"],
                                          "sample_submission.csv")
    submission = pd.read_csv(sample_submission_file, index_col=None)

    # test set에 대한 prediction
    file_names, preds = inference(model, test_dataloader, device, cfg)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, 
                                        "PredictionString" : ' '.join(str(e) for e in string.tolist())},
                                        ignore_index=True)

    # submission.csv로 저장
    submission_file = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"], 
                                   f"{cfg['SELECTED']['MODEL']}.csv")
    submission.to_csv(submission_file, index=False)
    print(f" >> Save Completed ... {submission_file}")
    

def main():
    cfg = get_cfg_from(get_args())
    
    wnb_run = wnb_init(cfg)
    print_ver_n_settings()
    pprint(cfg)
    
    df_train_categories_counts = get_df_train_categories_counts(cfg)
    sorted_df_train_categories_counts = add_bg_index_to(df_train_categories_counts)
    category_names = sorted_df_train_categories_counts["Categories"].to_list()
    
    _, _, test_dataloader = get_dataloaders(cfg, category_names)
    
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


if __name__ == "__main__":
    main()
