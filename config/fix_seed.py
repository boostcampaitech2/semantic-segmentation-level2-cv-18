import random

import numpy as np
import torch


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
    
