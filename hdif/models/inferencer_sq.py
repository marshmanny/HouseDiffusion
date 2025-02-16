import argparse
import collections
import io
import json
import os
import tempfile
import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch as th
import torchvision.transforms.functional as F
import webcolors
import drawsvg
import cairosvg

from omegaconf import OmegaConf
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos
from torchvision.utils import save_image
from tqdm.auto import tqdm

from hdif.utils.plotting.plot_from_feats import draw_from_batch
from hdif.datasets import DATASETS_REGISTRY, LOADERS_REGISTRY
from hdif.utils.inference_utils import add_dict_to_argparser
from hdif.trainers import TRAINERS_REGISTRY
from hdif.utils.seed import seed_setter
from hdif.models import INFERENCE_REGISTRY
from hdif.logging.metrics import mean_iou
from hdif.logging.metrics import estimate_graph
from hdif.logging.metrics import graph_edit_distance
from hdif.logging.metrics import rooms_overlap
from hdif.models.inferencer import BaseInferencer

def dict_to_device(dict: dict, device) -> dict:
    return {key: value.to(device) for key, value in dict.items()}


@INFERENCE_REGISTRY.add_to_registry("SquareInferencer")
class SquareInferencer(BaseInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ID_COLOR[-1] = "#E07190"
        
    
    def draw_svg_elements(self, draw_objs, polys, types, ID_COLOR, door_indices, resolution):
        # Первый проход: элементы, не являющиеся дверями
        fl = True
        for poly, c in zip(polys, types):
            if c in door_indices or c == 0:
                continue
            
            room_type = c
            color_hex = ID_COLOR[room_type]
            
            if fl:
                fl = False
                color_hex = ID_COLOR[-1]
            
            color_rgb = webcolors.hex_to_rgb(color_hex)
            stroke_color = webcolors.rgb_to_hex([int(x / 2) for x in color_rgb])
            # Рисуем заполненный полигон
            draw_objs['color'].append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                    fill=color_hex, fill_opacity=1.0,
                                                    stroke='black', stroke_width=1))
            # Рисуем контур
            draw_objs['main'].append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                fill='black', fill_opacity=0.0,
                                                stroke=stroke_color, stroke_width=0.5 * (resolution / 256)))
            draw_objs['second'].append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                    fill=color_hex, fill_opacity=1.0,
                                                    stroke=stroke_color, stroke_width=0.5 * (resolution / 256)))
            # Рисуем углы
            for corner in poly:
                draw_objs['main'].append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256),
                                                        fill=color_hex, fill_opacity=1.0,
                                                        stroke='gray', stroke_width=0.25))
                draw_objs['third'].append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256),
                                                        fill=color_hex, fill_opacity=1.0,
                                                        stroke='gray', stroke_width=0.25))
        # Второй проход: двери
        for poly, c in zip(polys, types):
            if c not in door_indices:
                continue
            room_type = c
            color_hex = ID_COLOR[room_type]
            color_rgb = webcolors.hex_to_rgb(color_hex)
            stroke_color = webcolors.rgb_to_hex([int(x / 2) for x in color_rgb])
            # Рисуем заполненный полигон для двери
            draw_objs['color'].append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                    fill=color_hex, fill_opacity=1.0,
                                                    stroke='black', stroke_width=1))
            # Рисуем контур двери
            draw_objs['main'].append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                fill='black', fill_opacity=0.0,
                                                stroke=stroke_color, stroke_width=0.5 * (resolution / 256)))
            draw_objs['second'].append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True,
                                                    fill=color_hex, fill_opacity=1.0,
                                                    stroke=stroke_color, stroke_width=0.5 * (resolution / 256)))
            # Рисуем углы двери
            for corner in poly:
                draw_objs['main'].append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256),
                                                        fill=color_hex, fill_opacity=1.0,
                                                        stroke='gray', stroke_width=0.25))
                draw_objs['third'].append(drawsvg.Circle(corner[0], corner[1], 2 * (resolution / 256),
                                                        fill=color_hex, fill_opacity=1.0,
                                                        stroke='gray', stroke_width=0.25))
        
        
    @seed_setter
    @torch.no_grad()
    def run(self):
        dataset_name = self.args.dataset
        
        dataset = DATASETS_REGISTRY[dataset_name]()
        
        loader = LOADERS_REGISTRY['infinite'](
            dataset,
            collate_fn=dataset.collate_fn,
            **{
                **self.config.loader.infinite.val,
                'infinite': False,
                'num_workers': 0,
                'pin_memory': False,
                'batch_size': 2,
            }
        )
        
        num_val_batches = np.ceil(
            len(dataset) / self.config.training.val_batch_size
        ).astype(int)
        
        p_join = lambda *args: os.path.join(*args)
        output_dir = str(p_join(self.args.output_dir))
        # output_dir = str(p_join(self.args.output_dir, f'{self.exp_name}'))
        
        Path(p_join(output_dir)).mkdir(parents=True, exist_ok=True)
        
        os.makedirs(f'{output_dir}/img', exist_ok=True)
        os.makedirs(f'{output_dir}/img/square/', exist_ok=True)
        # os.makedirs(f'{output_dir}/img/pred', exist_ok=True)
        # os.makedirs(f'{output_dir}/graphs', exist_ok=True)
        # os.makedirs(f'{output_dir}/graphs/gt', exist_ok=True)
        # os.makedirs(f'{output_dir}/graphs/pred', exist_ok=True)
        # os.makedirs(f'{output_dir}/metrics', exist_ok=True)
        
        graph_errors = []
        
        diffusion = self.trainer.diffusion
        model = self.trainer.model.eval()
        
        metrics = pd.DataFrame(columns=["id", "iou", "iou_gr", "eg_pred_in_eg_true", "overlap_iou_mean", "overlap_iou_median"])
        
        ii = 0
        
        for iter_num in tqdm(range(num_val_batches)):
            batch = next(loader)
            data_sample = batch["house"]
            if data_sample.shape[-1] == 2:  # Углы
                center, offsets = convert_to_center_offsets(data_sample)
                data_sample = torch.cat([center.unsqueeze(-1), offsets], dim=-1)

            model_kwargs = batch
            
            sample_fn = (
                diffusion.p_sample_loop if not self.model_args.use_ddim else diffusion.ddim_sample_loop
            )
            
            # batch['room_sizes'].shape, batch['room_indices'].shape
            # torch.Size([32, 100, 10]) torch.Size([32, 100, 32])
            
            ind_dim1, ind_dim2, ind_dim3 = torch.where(model_kwargs['room_indices'] == 1)
            mask_room = (ind_dim3 == 1)
            
            for size_i in range(10):
                iii = ii
                
                os.makedirs(f'{output_dir}/img/square/{size_i}', exist_ok=True)
                model_kwargs = dict_to_device(model_kwargs, "cuda")
                
                size_tensor = torch.zeros(10).double().cuda()
                size_tensor[size_i] = 1
                batch['room_sizes'][[0], [1]] = size_tensor
                batch['room_sizes'][ind_dim1[mask_room].long().cpu().numpy().tolist(),
                                    ind_dim2[mask_room].long().cpu().numpy().tolist()] = size_tensor

                sample_dict = sample_fn(
                    model,
                    data_sample.shape,
                    clip_denoised=self.model_args.clip_denoised,
                    model_kwargs=model_kwargs,
                    analog_bit=self.config.data.analog_bit,
                    return_dict=True
                )
                
                sample = sample_dict["samples"]
                timesteps = sample_dict["timesteps"]

                sample_gt = data_sample.unsqueeze(0)
                sample = convert_to_corners(sample[:, 0], sample[:, 1:])
                sample_gt = convert_to_corners(sample_gt[:, 0], sample_gt[:, 1:])

                
                # Timestep x batch index x num points x 2
                sample = sample.permute([0, 1, 3, 2]).cpu()
                sample_gt = sample_gt.permute([0, 1, 3, 2]).cpu()

                model_kwargs = dict_to_device(model_kwargs, "cpu")
                
                if self.config.data.analog_bit:
                    sample_gt = bin_to_int_sample(sample_gt)
                    sample = bin_to_int_sample(sample)

                sample_and_gt = {
                    "sample": sample.cpu(),
                    "timesteps": timesteps,
                    "sample_gt": sample_gt.cpu(),
                    "model_kwargs": dict_to_device(model_kwargs, "cpu"),
                    "id": model_kwargs["id"],
                }
                
                batch_size = sample_and_gt["sample"].shape[1]
                sample = sample_and_gt["sample"]
                sample_gt = sample_and_gt["sample_gt"]
                model_kwargs = sample_and_gt["model_kwargs"]
                
                pred_i_dict = self.get_floorplan_img(sample, model_kwargs)
                pred_gt_i_dict = self.get_floorplan_img(sample_gt, model_kwargs, is_syn=True)
                
                pred_i = pred_i_dict["images"]
                pred_gt_i = pred_gt_i_dict["images"]
                
                for i in range(len(pred_i)): 
                    id = model_kwargs["id"][i]
                    
                    if self.args.save_images:
                        # save imgs
                        pred_i[i].save(f'{output_dir}/img/square/{size_i}/{iii}.png')
                        # pred_gt_i[i].save(f'{output_dir}/img/square/gt/{ii}.png')
                    
                    # # get iou metric
                    # iou = mean_iou(pred_i[i], pred_gt_i[i], classes=self.CLASSES, color_rgb=self.color_rgb)
                    
                    # # get grapth metric
                    # graph_dict = estimate_graph(pred_i_dict["polys"][i], pred_i_dict["types"][i], model_kwargs[f'graph'][i], self.ID_COLOR, draw_graph=True)
                    
                    # if self.args.save_images:
                    #     graph_dict["G_true_img"].save(f'{output_dir}/graphs/gt/{ii}.png')
                    #     graph_dict["G_estimated_complete_img"].save(f'{output_dir}/graphs/pred/{ii}.png')
                    
                    # g_metrics = graph_edit_distance(graph_dict["G_gt_edges"], graph_dict["G_estimated_edges"], graph_dict["rooms"])
                    # o_metrics = rooms_overlap(pred_i_dict["polys"][i], graph_dict["rooms"])
                    # metrics.loc[len(metrics)] = [id.item(), iou] + [g_metrics[x] for x in ["iou_gr", "eg_pred_in_eg_true"]]  + [o_metrics[x] for x in ["overlap_iou_mean", "overlap_iou_median"]] # + [g_metrics[x] for x in ["pr", "rc", "f1"]] 
                    iii += 1
            ii += data_sample.shape[0]
            
            # metrics.to_csv(f'{output_dir}/metrics/all.csv', index=False)
            # metrics.columns = [x + "_mean" for x in metrics.columns]
            # pd.DataFrame(metrics.mean(axis=0)).to_csv(f'{output_dir}/metrics/summary.csv')
        

if __name__ == '__main__':
    model_args = Inferencer1.get_parser()
    args = model_args.parse_args()
    inferencer = Inferencer1(args)
