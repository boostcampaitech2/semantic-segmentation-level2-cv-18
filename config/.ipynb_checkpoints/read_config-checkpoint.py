import os
import argparse
import yaml

import torch


def print_ver_n_settings():
    print("-" * 30)
    print("pytorch version: {}".format(torch.__version__))
    print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    print("-" * 30)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg-yaml", type=str, default="./test.yaml", help="Choose your own cfg yaml."
    )

    args = parser.parse_args()

    return args


def cfg_check(cfg):
    selected_framework = cfg["SELECTED"]["FRAMEWORK"]
    assert selected_framework in cfg["FRAMEWORKS_AVAILABLE"]

    if selected_framework == "torchvision":
        assert cfg["SELECTED"]["MODEL"] in cfg["MODELS_AVAILABLE"][selected_framework]
    elif selected_framework == "segmentation_models_pytorch":
        assert cfg["SELECTED"]["MODEL_CFG"]["arch"] in cfg["DECODER_AVAILABLE"]
        assert cfg["SELECTED"]["MODEL_CFG"]["encoder_name"] in cfg["ENCODER_AVAILABLE"]


def get_cfg_from(args):
    with open(args.cfg_yaml, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    if cfg["EXPERIMENTS"]["LEARNING_RATE"]:
        cfg["EXPERIMENTS"]["LEARNING_RATE"] = float(cfg["EXPERIMENTS"]["LEARNING_RATE"])
    if cfg["SELECTED"]["OPTIMIZER_CFG"]["weight_decay"]:
        cfg["SELECTED"]["OPTIMIZER_CFG"]["weight_decay"] = float(
            cfg["SELECTED"]["OPTIMIZER_CFG"]["weight_decay"]
        )

    if not os.path.exists(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"]):
        os.mkdir(cfg["EXPERIMENTS"]["SAVED_DIR"]["BEST_MODEL"])
    if not os.path.exists(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"]):
        os.mkdir(cfg["EXPERIMENTS"]["SAVED_DIR"]["SUBMISSION"])

    cfg_check(cfg)

    return cfg
