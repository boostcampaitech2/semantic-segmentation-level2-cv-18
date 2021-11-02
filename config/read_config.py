import os
import argparse
import yaml

import torch

def print_ver_n_settings():
    """Print to console the version description."""
    print("-"*30)
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    print("-"*30)
    

def get_args():
    """By using parser, extract arguments.
    Return:
        args: argparse.Namespace 
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg-yaml", type=str, default="./test.yaml", help="Choose your own cfg yaml.")
    
    args = parser.parse_args()
    
    return args


def cfg_check(cfg):
    """Check the user's selection."""
    
    selected_framework = cfg["SELECTED"]["FRAMEWORK"]
    assert(selected_framework in cfg["FRAMEWORKS_AVAILABLE"])
    
    if selected_framework == "torchvision":
        assert(cfg["SELECTED"]["MODEL"] in cfg["MODELS_AVAILABLE"][selected_framework])
    elif selected_framework == "segmentation_models_pytorch":
        assert(cfg["SELECTED"]["MODEL_CFG"]["arch"] in cfg["DECODER_AVAILABLE"])
        assert(cfg["SELECTED"]["MODEL_CFG"]["encoder_name"] in cfg["ENCODER_AVAILABLE"])
    
    
    selected_criterion_framework = cfg["SELECTED"]["CRITERION"]["FRAMEWORK"]
    selected_criterion = cfg["SELECTED"]["CRITERION"]["USE"]
    assert(selected_criterion_framework in cfg["CRITERION_AVAILABLE"].keys())
    assert(selected_criterion in cfg["CRITERION_AVAILABLE"][selected_criterion_framework])
    
    tta_cfg = cfg["EXPERIMENTS"]["TTA"]
    if tta_cfg["TURN_ON"]:
        assert(any(tta_cfg["AVAILABLE_LIST"].values()))
    
    assert(cfg["EXPERIMENTS"]["LEARNING_RATE"] > 0)
    assert(cfg["EXPERIMENTS"]["NUM_EPOCHS"] >= cfg["EXPERIMENTS"]["VAL_EVERY"])
    
    
def get_cfg_from(args):
    """From arguments, extract configs."""
    with open(args.cfg_yaml, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if cfg["EXPERIMENTS"]["LEARNING_RATE"]:
        cfg["EXPERIMENTS"]["LEARNING_RATE"] = float(cfg["EXPERIMENTS"]["LEARNING_RATE"])
    
    if not os.path.exists(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"]):
        os.mkdir(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"])
    if not os.path.exists(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"]):
        os.mkdir(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"])
    
    cfg_check(cfg)
    
    return cfg

