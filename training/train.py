import os
import sys
import torch
# import pygraphviz

sys.path.append(".")

from hdif.trainers import TRAINERS_REGISTRY
import configs.train.arguments as arguments

def setup_seed(seed, cudnn_benchmark_off):
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark_off:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(config):
    setup_seed(config.exp.seed, config.exp.cudnn_benchmark_off)

    current_trainer = TRAINERS_REGISTRY[config.training.trainer](config)

    current_trainer.setup()

    current_trainer.train_loop()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # limit CPU usage
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENMP_NUM_THREADS"] = "1"
    torch.multiprocessing.set_start_method("spawn")

    config = arguments.load_config()
    
    if not config.log.wandb_chp:
        os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'
        print('name main')
    main(config)
