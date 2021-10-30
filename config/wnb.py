import wandb


def wnb_init(cfg):
    exp_cfg = cfg["EXPERIMENTS"]
    if exp_cfg["WNB_TURN_ON"]:
        run = wandb.init(**exp_cfg["WNB_INIT"])
        wandb.config.update(cfg)
        return run
    
