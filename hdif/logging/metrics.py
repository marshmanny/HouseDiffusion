import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos

import io
import webcolors
import drawsvg
import cairosvg
import collections

from PIL import Image
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from hdif.utils.plotting.pil_utils import get_concat_h
from hdif.utils.plotting.pil_utils import get_concat_h
from hdif.utils.plotting.plot_grid import add_names_to_header

from . import METRICS_REGISTRY

@METRICS_REGISTRY.add_to_registry("mIoU")
def mIoU(pred_mask, mask, classes, smooth=1e-10):
    """
    Computes the mean Intersection-over-Union between two masks;
    the predicted multi-class segmentation mask and the ground trutorch.
    """

    n_classes = len(classes)

    # make directly equipable when training (set grad off)
    with torch.no_grad():

        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for c in range(0, n_classes):  #loop over possible classes

            # compute masks per class
            true_class = pred_mask == c
            true_label = mask == c

            # when label does not exist in the ground truth, set to NaN
            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)

@METRICS_REGISTRY.add_to_registry("mean_iou")
def mean_iou(img_sub, img_ref, classes, color_rgb, smooth=1e-10):
    """
    Computes the mean Intersection-over-Union between two masks;
    the predicted multi-class segmentation mask and the ground trutorch.
    """

    # n_classes = len(classes)
    
    img_sub2 = np.zeros_like(img_sub)[:, :, 0]
    img_ref2 = np.zeros_like(img_ref)[:, :, 0]
    
    for cl in classes:
        mask = (np.asarray(img_sub) == color_rgb[cl]).all(-1)
        if mask.size != 0:
            img_sub2[mask] = cl
        
        mask = (np.asarray(img_ref) == color_rgb[cl]).all(-1)
        if mask.size != 0:
            img_ref2[mask] = cl
    
    img_sub2 = torch.tensor(img_sub2)
    img_ref2 = torch.tensor(img_ref2)
    
    pred_mask, mask = img_sub2, img_ref2

    # make directly equipable when training (set grad off)
    with torch.no_grad():

        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for c in classes:  #loop over possible classes

            # compute masks per class
            true_class = pred_mask == c
            true_label = mask == c

            # when label does not exist in the ground truth, set to NaN
            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)


def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array

def corners_to_poly_arr(sample_k_i, model_kwargs_i, prefix, resolution):
    polys, types = generate_polygons_and_types(sample_k_i, model_kwargs_i, prefix, resolution)
    arr = create_polygon([resolution, resolution], polys)
    return {"arr": arr, "polys": polys, "types": types}

def get_graph(g_true, ID_COLOR, draw_graph=True):
    # build true graph
    res_d = {}
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label
        if _type >= 0 and _type not in [11, 12]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    # add edges
    for k, m, l in g_true[1]:
        k = int(k)
        l = int(l)
        _type_k = g_true[0][k]
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [11, 12] and _type_l not in [11, 12]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        elif m > 0 and (_type_k==11 or _type_l==11):
            if _type_k==11:
                G_true.add_edges_from([(l, -1)])
            else:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    if draw_graph:
        fig = plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
        nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
                font_weight='bold', edgecolors=edgecolors, width=4.0, with_labels=False)
        plt.close('all')
        
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png', bbox_inches='tight')
        imgdata.seek(0)
        res_d['G_true_img'] = Image.open(imgdata)
        plt.close('all')
    
    return {"G_true": G_true, **res_d}

@METRICS_REGISTRY.add_to_registry("estimate_graph")
def estimate_graph(polys, nodes, G_gt, ID_COLOR, draw_graph):
    nodes = np.array(nodes)
    G_gt = G_gt[1-torch.where((G_gt == torch.tensor([0,0,0], device='cpu')).all(dim=1))[0]]
    G_gt_dict = get_graph([nodes, G_gt], ID_COLOR, draw_graph)
    G_gt = G_gt_dict["G_true"]
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    edge_labels = {}
    # add nodes
    for k, label in enumerate(nodes):
        _type = label
        if _type >= 0 and _type not in [11, 12]:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            linewidths.append(0.0)
    # add outside node
    G_estimated.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    # add node-to-door connections
    doors_inds = np.where((nodes == 11) | (nodes == 12))[0]
    rooms_inds = np.where((nodes != 11) & (nodes != 12))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:
                p1, p2 = polys[k], polys[l]
                p1, p2 = Polygon(p1), Polygon(p2)
                if not p1.is_valid:
                    p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
                if not p2.is_valid:
                    p2 = geom_factory(lgeos.GEOSMakeValid(p2._geom))
                
                u_sq = p1.union(p2).area
                iou = p1.intersection(p2).area / u_sq if u_sq else 0
                if iou > 0 and iou < 0.2:
                    doors_rooms_map[k].append((l, iou))
    # draw connections
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k] != 11:
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                edge_labels[(l1, l2)] = k
                G_estimated.add_edges_from([(l1, l2)])
        else:
            if len(_conn) > 0:
                l1 = _conn[0][0]
                edge_labels[(-1, l1)] = k
                G_estimated.add_edges_from([(-1, l1)])
    # add missed edges
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])
    # add edges colors 
    colors = []
    mistakes = 0
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')
    
    G_gt_edges = []
    G_estimated_edges = []
    
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l):
            G_gt_edges.append([k, l])
        if G_estimated.has_edge(k, l):
            G_estimated_edges.append([k, l])
    
    res_d = {
        **G_gt_dict,
        "doors": doors_inds,
        "rooms": rooms_inds,
        "G_gt_edges": G_gt_edges,
        "G_estimated_edges": G_estimated_edges,
    }
    
    if draw_graph:
        fig = plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
        weights = [4 for u, v in G_estimated_complete.edges()]
        nx.draw(G_estimated_complete, pos, edge_color=colors, linewidths=linewidths, edgecolors=edgecolors, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', width=weights, with_labels=False)
        
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png', bbox_inches='tight')
        imgdata.seek(0)
        res_d['G_estimated_complete_img'] = Image.open(imgdata)
        plt.close('all')
        
    return res_d

@METRICS_REGISTRY.add_to_registry("graph_edit_distance")
def graph_edit_distance(G_gt_edges, G_estimated_edges, rooms):
    result = dict()
    G_estimated_edges = [tuple(x) for x in G_estimated_edges]
    G_gt_edges = [tuple(x) for x in G_gt_edges]
    
    # y_pred = []
    # y_true = []
    
    # for i in rooms:
    #     for j in rooms:
    #         if (i, j) in G_estimated_edges:
    #             y_pred.append(1)
    #         else:
    #             y_pred.append(0)
            
    #         if (i, j) in G_gt_edges:
    #             y_true.append(1)
    #         else:
    #             y_true.append(0)
    
    # result["pr"], result["rc"], result["f1"], _ = precision_recall_fscore_support(y_true, y_pred)
    
    count = 0
    for edge in G_estimated_edges:
        if edge in G_gt_edges:
            count += 1

    G_estimated_edges_len = len(G_estimated_edges)
    result["eg_pred_in_eg_true"] = count / G_estimated_edges_len if G_estimated_edges_len else 0
    
    G_estimated_edges = G_estimated_edges + [(x[1], x[0]) for x in G_estimated_edges]
    G_gt_edges = G_gt_edges + [(x[1], x[0]) for x in G_gt_edges]
    
    G_estimated_edges = set(G_estimated_edges)
    G_gt_edges = set(G_gt_edges)
    
    result["iou_gr"] = len(G_estimated_edges & G_gt_edges) / len(G_estimated_edges | G_gt_edges)
    return result

@METRICS_REGISTRY.add_to_registry("rooms_overlap")
def rooms_overlap(polys, rooms):
    result = dict()
    result["overlap_iou"] = []
    
    for i in rooms:
        for j in rooms:
            if i > j:
                p1, p2 = polys[i], polys[j]
                p1, p2 = Polygon(p1), Polygon(p2)
                if not p1.is_valid:
                    p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
                if not p2.is_valid:
                    p2 = geom_factory(lgeos.GEOSMakeValid(p2._geom))
                
                u_sq = p1.union(p2).area
                iou = p1.intersection(p2).area / u_sq if u_sq else 0
                result["overlap_iou"].append(iou)
    
    result["overlap_iou_mean"] = np.mean(result["overlap_iou"])
    result["overlap_iou_median"] = np.median(result["overlap_iou"])
    return result

@METRICS_REGISTRY.add_to_registry("rplan_all_256", init=True)
class RPlanAllMetric256(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()
        self.resolution = resolution
        
        self.num_room_types = 14
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
        
    def generate_polygons_and_types(self, sample_k_i, model_kwargs_i, resolution):
        with open("expa/generate_polygons_and_types.txt", "a") as log_file:
            log_file.write("\n--- generate_polygons_and_types START ---\n")
            log_file.write(f"sample_k_i shape: {sample_k_i.shape if isinstance(sample_k_i, torch.Tensor) else type(sample_k_i)}\n")
            log_file.write(f"model_kwargs_i keys: {list(model_kwargs_i.keys())}\n")
            for key, value in model_kwargs_i.items():
                if isinstance(value, torch.Tensor):
                    log_file.write(f"{key}.shape: {value.shape}\n")
                else:
                    log_file.write(f"{key} is {type(value)}\n")
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
    
    def get_floor_poly_img(self, sample, model_kwargs):
        sample_i = sample[-1:]
        batch_size = sample.shape[1]
        images = []
        polyss = []
        typess = []

        with open("get_floor_poly_img.txt", "w") as log_file:
            log_file.write("--- GET_FLOOR_POLY_IMG INPUTS ---\n")
            log_file.write(f"sample shape: {sample.shape}\n")
            for key, value in model_kwargs.items():
                if isinstance(value, torch.Tensor):
                    log_file.write(f"{key} shape: {value.shape}\n")
                else:
                    log_file.write(f"{key} is {type(value)}\n")
        
        for i in range(batch_size):
            sample_i = sample[-1:]
            
            k = 0

            draw_objs = {
                'color': drawsvg.Drawing(self.resolution, self.resolution, displayInline=False)
            }
            draw_objs['color'].append(drawsvg.Rectangle(0, 0, self.resolution, self.resolution, fill='white'))
            
            # Генерируем полигоны и их типы
            sample_i_k_i = sample_i[k][i]
            model_kwargs_i = {key: value[i] for key, value in model_kwargs.items()}
            polys, types = self.generate_polygons_and_types(sample_i_k_i, model_kwargs_i, self.resolution)

            # Рисуем элементы на SVG
            self.draw_svg_elements(draw_objs, polys, types, self.ID_COLOR, self.door_indices, self.resolution)
            
            images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw_objs['color'].as_svg()))))
            polyss.append(polys)
            typess.append(types)
        
        return {"images": images, "polys": polyss, "types": typess}
        
    def _forward(self, sample_and_gt, cat_imgs=True):
        with open("expa/_forward_metrics.txt", "a") as f:
            f.write("--- _forward START ---\n")
            for key, value in sample_and_gt.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"{key} shape: {value.shape}\n")
                elif isinstance(value, dict):
                    f.write(f"{key} keys: {list(value.keys())}\n")
                else:
                    f.write(f"{key} is {type(value)}\n")
            f.write("--- _forward END ---\n\n")

        batch_size = sample_and_gt["sample"].shape[1]
        sample = sample_and_gt["sample"]
        sample_gt = sample_and_gt["sample_gt"]
        model_kwargs = sample_and_gt["model_kwargs"]

        center = sample[:, :, 0, :]  # [1, batch_size, 2]
        offsets = sample[:, :, 1:, :]  # [1, batch_size, num_points-1, 2]

        
        absolute_coords = center.unsqueeze(2) + offsets  # [1, batch_size, 100, 2]
        sample = absolute_coords  # [1, batch_size, 100, 2]


        
        pred_i_dict = self.get_floor_poly_img(sample, model_kwargs)
        pred_gt_i_dict = self.get_floor_poly_img(sample_gt, model_kwargs)
        
        pred_i = pred_i_dict["images"]
        pred_gt_i = pred_gt_i_dict["images"]
        
        result = {"imgs": {k: [] for k in ["fl_pred", "fl_gt", "g_gt", "g_pred"]},
                  "nums": {k: [] for k in ["iou", "iou_gr", "eg_pred_in_eg_true", "overlap_iou_mean", "overlap_iou_median"]}}
             
        for i in range(len(pred_i)):
            result["imgs"]["fl_gt"].append(pred_gt_i[i].resize((self.resolution, self.resolution)))
            result["imgs"]["fl_pred"].append(pred_i[i].resize((self.resolution, self.resolution)))
            
            # get iou metric
            result["nums"]["iou"] = mean_iou(pred_i[i], pred_gt_i[i], classes=self.CLASSES, color_rgb=self.color_rgb)
            
            # get grapth metric
            graph_dict = estimate_graph(pred_i_dict["polys"][i], pred_i_dict["types"][i], model_kwargs[f'graph'][i], self.ID_COLOR, draw_graph=True)
            result["imgs"]["g_gt"].append(graph_dict["G_true_img"].resize((self.resolution, self.resolution)))
            result["imgs"]["g_pred"].append(graph_dict["G_estimated_complete_img"].resize((self.resolution, self.resolution)))
            
            g_metrics = graph_edit_distance(graph_dict["G_gt_edges"], graph_dict["G_estimated_edges"], graph_dict["rooms"])
            [result["nums"][k].append(g_metrics[k]) for k in ["iou_gr", "eg_pred_in_eg_true"]]
            
            o_metrics = rooms_overlap(pred_i_dict["polys"][i], graph_dict["rooms"])
            [result["nums"][k].append(o_metrics[k]) for k in ["overlap_iou_mean", "overlap_iou_median"]]
                
        if cat_imgs:
            result["imgs"]["fl"] = []
            result["imgs"]["gr"] = []
            
            for img_pairs, group in zip([
                [result["imgs"]["fl_gt"], result["imgs"]["fl_pred"]],
                [result["imgs"]["g_gt"], result["imgs"]["g_pred"]]],
                ["fl", "gr"]):
                for im1, im2 in zip(*img_pairs):
                    img = get_concat_h(im1, im2)
                    img = add_names_to_header(img, ["gt", "pred"], SHAPE=self.resolution)
                    result["imgs"][group].append(img)
            
            for k in ["fl_pred", "fl_gt", "g_gt", "g_pred"]:
                del result["imgs"][k]
        return result
    
    def forward(self, sample_and_gt, cat_imgs=True):
        with open("forward_metrics.txt", "a") as f:
            f.write("--- forward START ---\n")
            for key, value in sample_and_gt.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"{key} shape: {value.shape}\n")
                elif isinstance(value, dict):
                    f.write(f"{key} keys: {list(value.keys())}\n")
                else:
                    f.write(f"{key} is {type(value)}\n")
            f.write("--- forward END ---\n\n")
        result = self._forward(sample_and_gt, cat_imgs)
        result = {**result["imgs"], **result["nums"]}
        return result

@METRICS_REGISTRY.add_to_registry("rplan_all_512", init=True)
class RPlanAllMetric512(RPlanAllMetric256):
    def __init__(self):
        super().__init__(resolution=512)


class CombinedMetric:
    def __init__(self, metrics):
        self.metrics = metrics
        self.metric_funcs = set()

    def __call__(
        self, data_dict, models
    ):
        metrics = collections.defaultdict(float)
        for m_name, m_params in self.metrics.items():
            if m_params.func in METRICS_REGISTRY.classes:
                self.metric_funcs.add(m_params.func)
                metric_name = f"{m_name}"
                inputs = {}
                for k, v in m_params.input_map.items():
                    try:
                        inputs[k] = data_dict.get(v, getattr(models, v, v))
                    except:
                        inputs[k] = v
                
                metric = METRICS_REGISTRY.classes[m_params.func](**inputs)
                if isinstance(metric, dict):
                    for sub_m_name, sub_m_value in metric.items():
                        metrics[sub_m_name] = sub_m_value
                else:
                    metrics[metric_name] = metric
            else:
                raise NotImplementedError(f"{m_params.func} loss is not implemented!")
        return metrics
