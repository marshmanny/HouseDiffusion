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

def dict_to_device(dict: dict, device) -> dict:
    return {key: value.to(device) for key, value in dict.items()}


@INFERENCE_REGISTRY.add_to_registry("BaseInferencer")
class BaseInferencer:
    """
    Inferencer1 implementation with hairstyle transfer interface
    """

    def __init__(self, args, model_args):
        self.args = args
        self.model_args = model_args
        
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            exp_folder = args.wandb_folder
            self.exp_name = os.path.basename(os.path.normpath(exp_folder))
            config_path = os.path.join(exp_folder, 'files', 'launch_config.yml')
            summary_path = os.path.join(exp_folder, 'files', 'wandb-summary.json')
            
            try:
                with open(summary_path, 'r') as file:
                    summary = json.load(file)
                
                epoch4load = summary['chp_info'][-1]['epoch']
            except:
                epoch4load = -1
            
            if args.epoch4load is not None:
                epoch4load = args.epoch4load
            
            config = OmegaConf.load(config_path)
            # config.pipeline.unerode_undilate = False
            # config.pipeline.dilate_mask = False
            # config.helpers.seg = {"model": "bisnet", "T": 1}
            # config.helpers.pp = "pretrained_models/PostProcess/pp_model.pth"
            # config.helpers.dilated_size2 = 5
            
            config.checkpoint.checkpoint4load = os.path.join(exp_folder, 'files', 'checkpoints')
            config.checkpoint.epoch4load = epoch4load
            # config.checkpoint.checkpoint4load = None
            # config.checkpoint.epoch4load = None
            
            trainer = TRAINERS_REGISTRY[config.training.trainer](config)
            trainer.run_dir = temp_dir
            trainer.setup_networks()
            trainer.setup_helpers()
            trainer.eval_mode()
            self.config = config
            self.trainer = trainer
        
        self.num_room_types = 14
        self.resolution = config.model.params.get("img_size", 256)
        
        self.ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                        13: '#785A67', 12: '#D3A2C7'}
        self.ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6,
                    "dining room": 7, "study room": 8,
                    "storage": 10, "front door": 15, "unknown": 16, "interior_door": 17}
        self.ROOM_NAMES = {v: k for k, v in self.ROOM_CLASS.items()}
        self.color_rgb = {k: np.array(webcolors.hex_to_rgb(v)) for k, v in self.ID_COLOR.items()}
        self.door_indices = [11, 12, 13]
        self.CLASSES = [x for x in list(self.ID_COLOR) if x not in self.door_indices]
    
    def draw_svg_elements(self, draw_objs, polys, types, ID_COLOR, door_indices, resolution):
        # Первый проход: элементы, не являющиеся дверями
        for poly, c in zip(polys, types):
            if c in door_indices or c == 0:
                continue
            room_type = c
            color_hex = ID_COLOR[room_type]
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
        
    def generate_polygons_and_types(self, sample_k_i, model_kwargs_i, prefix, resolution):
        polys = []
        types = []
        poly = []
        c = None
        for j, point in enumerate(sample_k_i):
            if model_kwargs_i[f'src_key_padding_mask'][j] == 1:
                continue
            point = point.cpu().numpy()
            if j == 0:
                poly = []
            if j > 0 and (model_kwargs_i[f'room_indices'][j] != model_kwargs_i[f'room_indices'][j - 1]).any():
                polys.append(poly)
                types.append(c)
                poly = []
            pred_center = False
            if pred_center:
                point = point / 2 + 1
                point = point * resolution // 2
            else:
                point = point / 2 + 0.5
                point = point * resolution
            poly.append((point[0], point[1]))
            c = np.argmax(model_kwargs_i[f'room_types'][j - 1].cpu().numpy())
        polys.append(poly)
        types.append(c)
        return polys, types
    
    def get_floorplan_img(self, sample, model_kwargs, is_syn=False):
        sample_i = sample[-1:]
        batch_size = sample.shape[1]
        images = []
        polyss = []
        typess = []
        
        for i in range(batch_size):
            sample_i = sample[-1:]
            prefix = 'syn_' if is_syn else ''
            
            k = 0
            # Создаем объекты drawsvg.Drawing
            draw_objs = {
                'main': drawsvg.Drawing(self.resolution, self.resolution, displayInline=False),
                'second': drawsvg.Drawing(self.resolution, self.resolution, displayInline=False),
                'third': drawsvg.Drawing(self.resolution, self.resolution, displayInline=False),
                'color': drawsvg.Drawing(self.resolution, self.resolution, displayInline=False)
            }
            draw_objs['main'].append(drawsvg.Rectangle(0, 0, self.resolution, self.resolution, fill='black'))
            draw_objs['second'].append(drawsvg.Rectangle(0, 0, self.resolution, self.resolution, fill='black'))
            draw_objs['third'].append(drawsvg.Rectangle(0, 0, self.resolution, self.resolution, fill='black'))
            draw_objs['color'].append(drawsvg.Rectangle(0, 0, self.resolution, self.resolution, fill='white'))
            
            # Генерируем полигоны и их типы
            sample_i_k_i = sample_i[k][i]
            #change
            if sample_i_k_i.shape[-1]== 2:  # Если уже в нужном формате, не конвертируем
                sample_i_k_i = convert_to_corners(sample_i_k_i[:, 0], sample_i_k_i[:, 1:])

            model_kwargs_i = {key: value[i] for key, value in model_kwargs.items()}
            polys, types = self.generate_polygons_and_types(sample_i_k_i, model_kwargs_i, prefix, self.resolution)

            # Рисуем элементы на SVG
            self.draw_svg_elements(draw_objs, polys, types, self.ID_COLOR, self.door_indices, self.resolution)
            
            images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw_objs['color'].as_svg()))))
            polyss.append(polys)
            typess.append(types)
        
        return {"images": images, "polys": polyss, "types": typess}
    
    
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
                # 'batch_size': 2,
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
        os.makedirs(f'{output_dir}/img/gt', exist_ok=True)
        os.makedirs(f'{output_dir}/img/pred', exist_ok=True)
        os.makedirs(f'{output_dir}/graphs', exist_ok=True)
        os.makedirs(f'{output_dir}/graphs/gt', exist_ok=True)
        os.makedirs(f'{output_dir}/graphs/pred', exist_ok=True)
        os.makedirs(f'{output_dir}/metrics', exist_ok=True)
        
        graph_errors = []
        
        diffusion = self.trainer.diffusion
        model = self.trainer.model.eval()
        ii = 0
        
        metrics = pd.DataFrame(columns=["id", "iou", "iou_gr", "eg_pred_in_eg_true", "overlap_iou_mean", "overlap_iou_median"])
    
        for iter_num in tqdm(range(num_val_batches)):
            batch = next(loader)
            data_sample = batch["house"]

            # Проверяем, что данные уже в формате (центр + смещения)
            if data_sample.shape[-1] == 2:  # Данные хранятся в формате углов -> конвертируем
                center, offsets = convert_to_center_offsets(data_sample)
                data_sample = torch.cat([center.unsqueeze(-1), offsets], dim=-1)

            model_kwargs = batch

            
            sample_fn = (
                diffusion.p_sample_loop if not self.model_args.use_ddim else diffusion.ddim_sample_loop
            )

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

            # Преобразуем (центр + смещения) обратно в углы
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
                    pred_i[i].save(f'{output_dir}/img/pred/{ii}.png')
                    pred_gt_i[i].save(f'{output_dir}/img/gt/{ii}.png')
                pred_i_dict["polys"][i] = convert_to_corners(pred_i_dict["polys"][i][:, 0], pred_i_dict["polys"][i][:, 1:])

                
                # get iou metric
                iou = mean_iou(pred_i[i], pred_gt_i[i], classes=self.CLASSES, color_rgb=self.color_rgb)
                
                # get grapth metric
                graph_dict = estimate_graph(pred_i_dict["polys"][i], pred_i_dict["types"][i], model_kwargs[f'graph'][i], self.ID_COLOR, draw_graph=True)
                
                if self.args.save_images:
                    graph_dict["G_true_img"].save(f'{output_dir}/graphs/gt/{ii}.png')
                    graph_dict["G_estimated_complete_img"].save(f'{output_dir}/graphs/pred/{ii}.png')
                
                g_metrics = graph_edit_distance(graph_dict["G_gt_edges"], graph_dict["G_estimated_edges"], graph_dict["rooms"])
                o_metrics = rooms_overlap(pred_i_dict["polys"][i], graph_dict["rooms"])
                metrics.loc[len(metrics)] = [id.item(), iou] + [g_metrics[x] for x in ["iou_gr", "eg_pred_in_eg_true"]]  + [o_metrics[x] for x in ["overlap_iou_mean", "overlap_iou_median"]] # + [g_metrics[x] for x in ["pr", "rc", "f1"]] 
                ii += 1
        
        metrics.to_csv(f'{output_dir}/metrics/all.csv', index=False)
        metrics.columns = [x + "_mean" for x in metrics.columns]
        pd.DataFrame(metrics.mean(axis=0)).to_csv(f'{output_dir}/metrics/summary.csv')
        
    def get_parser():
        parser = argparse.ArgumentParser(description='Inferencer1')
        # Arguments
        parser.add_argument('--device', type=str, default='cuda')

        # Inferencer1 setting
        defaults = dict(
            clip_denoised=True,
            use_ddim=False,
            draw_graph=False,
            save_svg=False,
            save_gif=False,
        )
        
        add_dict_to_argparser(parser, defaults)
        # parser.add_argument('--rotate_checkpoint', type=str, default='pretrained_models/Rotate/rotate_best.pth')
        # parser.add_argument('--pp_checkpoint', type=str, default='pretrained_models/PostProcess/pp_model.pth')
        
        return parser

if __name__ == '__main__':
    model_args = Inferencer1.get_parser()
    args = model_args.parse_args()
    inferencer = Inferencer1(args)
