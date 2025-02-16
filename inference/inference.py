import argparse
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

from torchvision.utils import save_image
from tqdm.auto import tqdm
import wandb

import string
import numpy as np
from hdif.models import INFERENCE_REGISTRY

def main(model_args, args):
    align = INFERENCE_REGISTRY[args.inferencer](args, model_args)
    align.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--inferencer', type=str, default='Inferencer1')
    
    # Arguments for a set of experiments
    parser.add_argument('--dataset', type=str, default="rplan_inf",
                        help='Dataset for inference')
    parser.add_argument('--fids', '--names-list', nargs='+', default=[])
    parser.add_argument('--wandb_folder', type=str, default='wandb/foler_exp')
    parser.add_argument('--epoch4load', nargs='?', type=int, const=-1, help='')
    
    parser.add_argument('--save_images', action='store_true', help='')
    parser.add_argument('--save_fids', action='store_true', help='')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')
    parser.add_argument('--output_filename', type=str, default='metrics.csv', help='')
    
    args, unknown1 = parser.parse_known_args()
    
    model_parser = INFERENCE_REGISTRY[args.inferencer].get_parser()
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_)
        model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)

    main(model_args, args)
