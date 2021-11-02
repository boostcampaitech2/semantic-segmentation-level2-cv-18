import os
from tqdm import tqdm
from pprint import pprint
from functools import reduce
import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import albumentations as A

import torch
import torch.nn.functional as F

from data.dataloader import (
    get_dataloaders
)

from config.read_config import (
    print_ver_n_settings, get_args, get_cfg_from
)
from config.wnb import(
    wnb_init
)
from config.fix_seed import (
    fix_seed_as
)

from util.eda import (
    get_df_train_categories_counts, add_bg_index_to
)
from util.ploting import (
    plot_examples
)
from util.tta import (
    get_tta_list
)

from train import (
    get_trainable_model, get_model_file_name, get_model_inference
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def get_model_inference_considered_tta(cfg, model, images):
    frame_selected = cfg["SELECTED"]["FRAMEWORK"]
    outputs = get_model_inference(cfg, model, images)
    
    tta_list = get_tta_list(cfg)
    if cfg["EXPERIMENTS"]["TTA"]["TURN_ON"]:
        dict_a_tta2transformed_images = {a_tta : a_tta(images) for a_tta in tta_list}
        
        for a_tta, transformed_images in dict_a_tta2transformed_images.items():
            tta_outputs = a_tta(get_model_inference(cfg, model, transformed_images))
            outputs += tta_outputs
        
        outputs /= len(dict_a_tta2transformed_images) + 1
        
    return outputs
    
    
def get_trained_model(cfg, device, fold:int=None):
    model = get_trainable_model(cfg)
    # best model 저장된 경로
    
    model_path = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"], 
                              get_model_file_name(cfg, fold))
    
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    return model


# 제공받은 baseline -> wandb 등 custom
def inference_one(model, test_dataloader, device, cfg):
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
    print('Start prediction ... ')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_dataloader)):
                
            # inference (512 x 512)
            outs = get_model_inference_considered_tta(cfg, model, torch.stack(imgs).to(device))
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
    
    if cfg["EXPERIMENTS"]["WNB"]["TURN_ON"]:
        plot_examples(model=model,
                      cfg=cfg,
                      device=device,
                      mode="test", 
                      batch_id=0, 
                      num_examples=cfg["EXPERIMENTS"]["BATCH_SIZE"],
                      dataloader=test_dataloader)
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array
    

# 제공받은 baseline -> wandb 등 custom
def inference_kfold(models, test_dataloader, device, cfg):
    """By using trained model, infer all images in test_dataloader
    Args:
        models: A list of several trained models
        test_dataloader : DataLoader
        device: inference place
        
    Returns:
        file_names: the file names for all images in test_dataloader
        preds_array: the model prediction for all images in test_dataloader
    """
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction (kfold) ...')
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_dataloader)):
            
            # inference (512 x 512)
            final_outs = None
            for model in models:
                model = model.to(device)
                model.eval()
                
                outs = get_model_inference_considered_tta(cfg, model, torch.stack(imgs).to(device))
                final_outs = outs if final_outs is None else final_outs + outs
            final_outs /= len(models)
            
            final_outs = F.softmax(final_outs, dim=1)
            oms = torch.argmax(final_outs.squeeze(), dim=1).detach().cpu().numpy()
            
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
    
    if cfg["EXPERIMENTS"]["WNB"]["TURN_ON"]:
        plot_examples(model=model,
                      cfg=cfg,
                      device=device,
                      mode="test", 
                      batch_id=0, 
                      num_examples=cfg["EXPERIMENTS"]["BATCH_SIZE"], 
                      dataloader=test_dataloader)
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array
    
    
def inference(test_dataloader, device, cfg):
    if not cfg["EXPERIMENTS"]["KFOLD"]["TURN_ON"]:
        model_trained = get_trained_model(cfg, DEVICE)
        
        return inference_one(model_trained, 
                             test_dataloader, 
                             device, 
                             cfg)
    else:
        models_trained = [get_trained_model(cfg, DEVICE, fold) for fold in range(cfg["EXPERIMENTS"]["KFOLD"]["NUM_FOLD"])]
            
        return inference_kfold(models_trained, 
                               test_dataloader, 
                               device, 
                               cfg)


def get_submission_file_name(cfg):
    submission_file = ""
    file_name_components = []
    seleceted_framework = cfg["SELECTED"]["FRAMEWORK"]
    
    if seleceted_framework == "torchvision":
        file_name_components.append(cfg['SELECTED']['MODEL'])
        
    elif seleceted_framework == "segmentation_models_pytorch":
        arch_name = cfg['SELECTED']['MODEL_CFG']['arch']
        enc_name = cfg['SELECTED']['MODEL_CFG']['encoder_name']
        enc_weights_name = cfg['SELECTED']['MODEL_CFG']['encoder_weights']
        
        file_name_components.append(arch_name)
        file_name_components.append(enc_name)
        file_name_components.append(enc_weights_name)
        
    if cfg["EXPERIMENTS"]["KFOLD"]["TURN_ON"]:
        file_name_components.append(f"{cfg['EXPERIMENTS']['KFOLD']['NUM_FOLD']}fold")
    
    if cfg["EXPERIMENTS"]["TTA"]["TURN_ON"]:
        file_name_components.append("TTA")
    
    file_name = "_".join(file_name_components) + ".csv"
    submission_file = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"], file_name)
    return submission_file
    

# 제공받은 baseline -> cfg 적용
def create_submission(test_dataloader, device, cfg):
    
    # sample_submisson.csv 열기
    sample_submission_file = os.path.join(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"],
                                          "sample_submission.csv")
    submission = pd.read_csv(sample_submission_file, index_col=None)

    # test set에 대한 prediction
    file_names, preds = inference(test_dataloader, device, cfg)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, 
                                        "PredictionString" : ' '.join(str(e) for e in string.tolist())},
                                        ignore_index=True)

    # submission.csv로 저장
    submission_file = get_submission_file_name(cfg)
    
    submission.to_csv(submission_file, index=False)
    print(f" >> Save Completed ... {submission_file}")
    

def main():
    cfg = get_cfg_from(get_args())
    fix_seed_as(cfg["SEED"])
    
    wnb_run = wnb_init(cfg)
    print_ver_n_settings()
    pprint(cfg)
    
    df_train_categories_counts = get_df_train_categories_counts(cfg)
    sorted_df_train_categories_counts = add_bg_index_to(df_train_categories_counts)
    category_names = sorted_df_train_categories_counts["Categories"].to_list()
    
    _, _, test_dataloader = get_dataloaders(cfg, category_names)
    
    create_submission(test_dataloader=test_dataloader, 
                      device=DEVICE,
                      cfg=cfg)
    
    if wnb_run is not None:
        wnb_run.finish()


if __name__ == "__main__":
    main()

