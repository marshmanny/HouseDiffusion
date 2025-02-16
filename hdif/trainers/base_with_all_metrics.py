import os
import copy
import torch
import torch.utils.data
import torch.nn.functional

import joblib
import hdif.utils.log_utils as log_utils
import omegaconf
import collections
import sys
from argparse import Namespace
from torchvision import transforms as T
from hdif.utils.plotting.plot_from_feats import plot_predictions
import numpy as np

from hdif.utils.nn import update_ema
from hdif.utils.resample import create_named_schedule_sampler
from hdif.utils.bicubic import BicubicDownSample
from hdif.utils.funcs import DilatedMask
from hdif.utils.image_utils import DilateErosion
from hdif.utils.train import image_grid, _LegacyUnpickler
from hdif.utils import trainer_utils
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import hdif.losses.losses as losses
import hdif.logging.metrics as metrics
from hdif.utils.trainer_utils import dict_to_device 

from hdif.models import MODELS_REGISTRY
from hdif.datasets import DATASETS_REGISTRY
from hdif.datasets import LOADERS_REGISTRY
from hdif.trainers import TRAINERS_REGISTRY
from hdif.logging import METRICS_REGISTRY


@TRAINERS_REGISTRY.add_to_registry(name="base_wi_all_metrics")
class BaseTrainerWiAllMetrics:
    def __init__(self, config):
        # initialized in setup_logger()
        self.logger = None
        self.run_dir = None

        # initialized in setup_dataset()
        self.dataset = None

        # initialized in setup_loaders()
        self.loaders = None

        # initialized in setup_networks()
        self.model = None
        self.gen_obj = None
        self.e4e = None
        self.fs_enc = None
        self.seg = None
        self.discs = None
        self.nets = None
        self.checkpoint_dir = None

        # initialized in setup_optimizers()
        self.optim_model = None
        self.optim_discs = None
        self.optims = None

        # initialized in setup_losses()
        self.combined_model_loss = None
        self.combined_disc_loss = None

        # initialized in setup_num_iters()
        self.num_iters = None
        self.num_val_iters = None

        # initialized in setup_metrics()
        self.fid = None

        self.config = config

    def setup_logger(self):
        config_for_logger = omegaconf.OmegaConf.to_container(self.config)
        config_for_logger["PID"] = os.getpid()
        exp_logger = log_utils.WandbLogger(
            save_code=self.config.log.save_code,
            omega_config=self.config,
            project=self.config.exp.project,
            name=self.config.exp.name,
            dir=self.config.exp.root,
            tags=tuple(self.config.exp.tags) if self.config.exp.tags else None,
            notes=self.config.exp.notes,
            mode=self.config.exp.mode,
            config=config_for_logger,
        )
        self.run_dir = exp_logger.run_dir
        console_logger = log_utils.ConsoleLogger(self.config.exp.name)
        self.logger = log_utils.LoggingManager(exp_logger, console_logger)
        self.model_logger = log_utils.ModelLogging(self.logger)
        self.chp_logger = log_utils.CheckpointLogger(self.config.checkpoint)

    def setup_dataset(self):
        self.dataset = dict()
        for data_type in ["train", "val", "test"]:
            dataset_name = self.config.data.name
            dataset = DATASETS_REGISTRY[dataset_name](
                **self.config.dataset[dataset_name][data_type]
            )
            self.dataset[data_type] = dataset
            self.config.dataset[dataset_name][data_type]["len"] = len(dataset)
        
    def setup_loaders(self):
        self.loaders = dict()
        loader_type = self.config.data.loader
        loader_args = self.config.loader[loader_type]
        
        for data_type in ["train", "val", "test"]:
            self.loaders[data_type] = LOADERS_REGISTRY[loader_type](
                self.dataset[data_type],
                **loader_args[data_type],
                collate_fn=self.dataset[data_type].collate_fn if hasattr(self.dataset[data_type], 'collate_fn') \
                    else torch.utils.data.default_collate
            )

    def setup_networks(self):
        print("\nLoading Model")
        self.model = MODELS_REGISTRY[self.config.model.model](
            **self.config.model.params
        )
        
        if hasattr(self.model, '_input_shape'):
            log_utils.print_network(
                self.model,
                f"model_{self.config.model.model}",
                torch.randn(self.model._input_shape),
                self.logger,
                self.config.log.log_complexity,
            )
            
        device = self.config.training.device
        
        if self.config.training.ema.rate is not None:
            ema_rate = self.config.training.ema.rate
            self.ema_rate = (
                [ema_rate]
                if isinstance(ema_rate, float)
                else [float(x) for x in ema_rate.split(",")]
            )
            self.ema_models = [
                    copy.deepcopy(self.model)
                    for _ in range(len(self.ema_rate))
                ]
            for ema_model in self.ema_models:
                for p in ema_model.parameters():
                    p.detach_()
                ema_model = ema_model.to(device)
            
        self.nets = dict(model=self.model)
        self.to(device)
        
        if self.config.checkpoint.checkpoint4load is not None:
            log_utils.restore_checkpoint_from_dir(
                self.config.checkpoint.checkpoint4load,
                self.config.checkpoint.epoch4load,
                self.nets,
                self.config.training.device,
            )
        
        self.diffusion = MODELS_REGISTRY[self.config.diffusion.diffusion](
            **self.config.diffusion.params
        )
        
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir)

    def to(self, device):
        for net in self.nets.values():
            net.to(device)

    def train_mode(self):
        for net in self.nets.values():
            net.train()
        self.stage = "train"

    def eval_mode(self):
        for net in self.nets.values():
            net.eval()
        self.stage = "eval"
    
    def test_mode(self):
        for net in self.nets.values():
            net.eval()
        self.stage = "test"

    def setup_optimizers(self):
        self.optim_model = getattr(torch.optim, self.config.model.opt)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.config.model.opt_params,
        )
        self.optims = dict(optim_model=self.optim_model)

    def setup_schedulers(self):
        self.schedulers = dict()

    def setup_losses(self):
        self.combined_model_loss = losses.CombinedLoss(self.config.model.losses)
    
    def setup_helpers(self):
        self.downsample_256 = BicubicDownSample(factor=4, cuda=self.config.training.device == "cuda")
        self.schedule_sampler = create_named_schedule_sampler(self.config.helpers.schedule_sampler, self.diffusion)
    
    def setup_metrics(self):
        self.combined_metric = dict()
        for data_type in ["train", "val", "test"]:
            self.combined_metric[data_type] = metrics.CombinedMetric(self.config.model.metrics[data_type])

    def setup_num_iters(self):
        num_batches = len(self.dataset["train"]) // self.config.training.batch_size
        self.num_iters = (
            self.config.training.num_iters
            if self.config.training.num_iters
            else num_batches
        )
        num_val_batches = (
            len(self.dataset["val"]) // self.config.training.val_batch_size
        )
        self.num_val_iters = (
            self.config.training.num_val_iters
            if self.config.training.num_val_iters
            else num_val_batches
        )
        num_test_batches = (
            len(self.dataset["test"]) // self.config.training.test_batch_size
        )
        self.num_test_iters = (
            self.config.training.num_test_iters
            if self.config.training.num_test_iters
            else num_test_batches
        )
        self.train_iter = 0

    def setup(self):
        self.setup_dataset()
        self.setup_loaders()
        self.setup_logger()
        self.setup_networks()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_num_iters()
        self.setup_metrics()
        self.setup_helpers()
    
    @torch.inference_mode()
    def compute_metrics(self, images, epoch_info, epoch_num, log_event='val'):
        result = dict()
        def compute_fid_datasets(images):
            self.fid.reset()

            fake_dataloader = DataLoader(TensorDataset(images), batch_size=128)
            for batch in fake_dataloader:
                batch = batch[0].to(self.config.training.device)
                self.fid.update(batch, real=False)

            return self.fid.compute()
        
        if self.fid is not None:
            result["metrics/fid"] = compute_fid_datasets(images)
        result = {f"{log_event}/{k}": v for k, v in result.items()}
        result["metrics/lr_model"] = self.optim_model.param_groups[0]['lr']
        
        self.logger.log_iter(self.train_iter, epoch_num, result)
        epoch_info.update(result)
        return epoch_info
    
    def check_is_image(self, tensor):
        v_shape = tensor.shape
        if len(v_shape) == 4 and tuple(v_shape[-2:]) in [(256, 256),
                                                         (512, 512),
                                                         (1024, 1024)]:
            return True
        return False
    
    def reshape_images(self, image_dict, result_dict):
        for k, v in image_dict.items():        
            if self.check_is_image(v):
                result_dict[k] = [] if k not in result_dict else result_dict[k]
                result_dict[k].append(self.downsample_256(v).cpu() if v.shape[-2:] == (1024, 1024) else v.cpu())
        return result_dict
    
    def generate_log_images(self, reshaped_images):
        return []
    
    def model_loss(self, data_dict, iter_info, log_event="train"):
        log_event += "_model"
        
        losses = self.combined_model_loss(data_dict, self.train_iter, self, self.stage)
        iter_info.update({f"{log_event}/{k}": v for k, v in losses.items()})
        return losses["total"]
    
    def model_metrics(self, data_dict, iter_info, log_event="train"):
        metrics = self.combined_metric[log_event](data_dict, self)
        
        iter_info.update({f"{log_event}/{k}": v for k, v in metrics.items()})
        return metrics
    

    def forward_model(self, data_dict):
        dd = data_dict
        result = dict()
        
        center,offsets  = dd['center'], dd['offsets'] 
        center = center.unsqueeze(1)

        combined_input = torch.cat([center, offsets], dim=1)  
        combined_input = combined_input.permute(0, 2, 1)

        t, result["batch_weights"] = self.schedule_sampler.sample(combined_input.shape[0], self.config.training.device)
        result["target"] = torch.randn_like(combined_input)

        src_mask = dd['src_key_padding_mask']  # [512, 100]
        center_mask = torch.ones((src_mask.shape[0], 1), device=src_mask.device)
        full_mask = torch.cat([center_mask, src_mask], dim=1)
        result["tmp_mask"] = 1 - full_mask

        x_t = self.diffusion.q_sample(combined_input, t, noise=result["target"])

        xtalpha = trainer_utils._extract_into_tensor(self.diffusion.sqrt_recip_alphas_cumprod, t, x_t.shape).permute([0, 2, 1])
        epsalpha = trainer_utils._extract_into_tensor(self.diffusion.sqrt_recipm1_alphas_cumprod, t, x_t.shape).permute([0, 2, 1])

        result["model_output_dec"], result["model_output_bin"] = self.model(
            x_t,
            self.diffusion._scale_timesteps(t),
            xtalpha=xtalpha,
            epsalpha=epsalpha,
            **dd
        )

        if not self.config.data.analog_bit:
            def dec2bin(xinp, bits):
                mask = 2 ** torch.arange(bits - 1, -1, -1).to(xinp.device, xinp.dtype)
                return xinp.unsqueeze(-1).bitwise_and(mask).ne(0).float()

            result["bin_target"] = combined_input.detach()
            result["bin_target"] = (result["bin_target"] / 2 + 0.5)  # -> [0,1]
            result["bin_target"] = result["bin_target"] * self.diffusion.img_size  # -> [0, 256]

            result["bin_target"] = dec2bin(result["bin_target"].permute([0, 2, 1]).round().int(), self.diffusion.num_bits)
            result["bin_target"] = result["bin_target"].reshape(
                [result["target"].shape[0], result["target"].shape[2], 2 * self.diffusion.num_bits]
            ).permute([0, 2, 1])

            result["t_weights"] = (t < 10).cuda().unsqueeze(1).unsqueeze(2)
            result["t_weights"] = result["t_weights"] * (result["t_weights"].shape[0] / max(1, result["t_weights"].sum()))
            result["bin_target"][result["bin_target"] == 0] = -1

        result["ones_like_target"] = torch.ones_like(result["target"])
        return result

    
    def sample(self, shape, cond_batch):

        result = self.diffusion.p_sample_loop(
            self.model,
            shape,
            clip_denoised=True,
            model_kwargs=cond_batch,
            analog_bit=self.config.data.analog_bit,
            return_every_nth=10,
            return_dict=True
        )
        return result
    
    
    def sample_with_gt(self, data_sample_gt, model_kwargs):

        sample_dict = self.sample(data_sample_gt.shape, model_kwargs)

        sample = sample_dict["samples"]
        timesteps = sample_dict["timesteps"]

        sample_gt = data_sample_gt.unsqueeze(0)
        print(f'sample_with_gt{data_sample_gt.shape}')

        with open("sample_with_gt.txt", "a") as f:
            f.write("--- sample_with_gt START ---\n")
            f.write(f"data_sample_gt shape: {data_sample_gt.shape}\n")
            f.write(f"sample_dict['samples'] shape: {sample.shape}\n")
            f.write(f"sample_dict['timesteps'] type: {type(timesteps)}\n")

        sample = sample.permute([0, 1, 3, 2]).cpu()
        sample_gt = sample_gt.permute([0, 1, 3, 2]).cpu()

        with open("expa/sample_with_gt.txt", "a") as f:
            f.write(f"sample shape after permute: {sample.shape}\n")
            f.write(f"sample_gt shape after permute: {sample_gt.shape}\n")

        model_kwargs = dict_to_device(model_kwargs, "cpu")
        s_gt = model_kwargs['house']
        s_gt = s_gt.permute(0, 2, 1)  # [16, 100, 2]
        s_gt = s_gt.unsqueeze(0)  

        sample_and_gt = {
            "sample": sample.cpu(),
            "timesteps": timesteps,
            "sample_gt": s_gt.cpu(),
            "model_kwargs": dict_to_device(model_kwargs, "cpu"),
            "id": model_kwargs["id"],
        }
        with open("expa/sample_with_gt.txt", "a") as f:
            f.write(f"sample_and_gt keys: {list(sample_and_gt.keys())}\n")
            for key, value in sample_and_gt.items():
                if isinstance(value, torch.Tensor):
                    f.write(f"{key} shape: {value.shape}\n")
                else:
                    f.write(f"{key} type: {type(value)}\n")
            f.write("--- sample_with_gt END ---\n\n")

        return {"sample_and_gt": sample_and_gt}

    
    def model_train_step(self, batch, iter_info):
        self.optim_model.zero_grad()
        outputs = self.forward_model(batch)    

        
        loss = self.model_loss({**batch, **outputs}, iter_info)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.model.clip_grad_norm)
        self.optim_model.step()
        
        if self.train_iter % self.config.training.ema.every_iter == 0 and self.config.training.ema.rate is not None:
            self.update_ema()
        return outputs
        
    def update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_models):
            update_ema(params.parameters(), self.model.parameters(), rate=rate)
    
    def update_lr(self):
        if self.train_iter % self.config.training.upd_lr_iter == 0:
            lr = self.optim_model.param_groups[0]['lr'] * (0.1**(self.train_iter//self.config.training.upd_lr_iter))
            
            for param_group in self.optim_model.param_groups:
                param_group["lr"] = lr
    
    def train_epoch(self, epoch_num, epoch_info):
        print("TRAIN EPOCH")
        iter_info = log_utils.StreamingVals()

        for iter_num in tqdm(range(self.num_iters)):
            batch = next(self.loaders["train"])

            outputs = self.model_train_step(batch, iter_info)
            inputs_outputs = {**batch, **outputs}         
            
            # in iter_info mean values with previous history are calculated
            self.model_metrics(inputs_outputs, iter_info, log_event="train")

            self.train_iter += 1
            self.update_lr()
        
        epoch_info.update(iter_info)    
        return epoch_info

    @torch.inference_mode()
    def validation_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingVals()
        
        for iter_num in tqdm(range(self.num_val_iters)):
            batch = next(self.loaders["val"])
            outputs = self.forward_model(batch)
            inputs_outputs = {**batch, **outputs}
            self.model_loss(inputs_outputs, iter_info, log_event="val")
            self.model_metrics(inputs_outputs, iter_info, log_event="val")
        
        iter_info["training/lr_model"] = self.optim_model.param_groups[0]['lr']
        epoch_info.update(iter_info)
        return epoch_info
    

    @torch.inference_mode()
    def test_epoch(self, epoch_num, epoch_info):
        iter_info = log_utils.StreamingVals()


        for iter_num in tqdm(range(self.num_test_iters)):
            batch = next(self.loaders["test"])
            with open("test_epoch.txt", "a") as f:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        f.write(f"{key}.shape: {value.shape}\n")
                    else:
                        f.write(f"{key} is {type(value)}\n")
            center = batch['center'].unsqueeze(1)  # [B, 1, 2] 
            offsets = batch['offsets']  # [B, N, 2]
            combined_input = torch.cat([center, offsets], dim=1).permute(0, 2, 1)  # [B, 2, N+1]
            
            outputs = self.sample_with_gt(combined_input, batch)
            inputs_outputs = {**batch, **outputs}
            with open("THIS_IMPUT.log", "a") as f:
                f.write("INPUTS_OUTPUTS SHAPES:\n")
                for key, value in inputs_outputs.items():
                    if isinstance(value, torch.Tensor):
                        f.write(f"{key}.shape: {value.shape}\n")
                    else:
                        f.write(f"{key} is {type(value)}\n")
                f.write("--- ITERATION END ---\n\n")

            self.model_metrics(inputs_outputs, iter_info, log_event="test")
        
        epoch_info.update(iter_info)

        return epoch_info

    
    def train_loop(self):
        self.start_epoch = 0
        self.setup_schedulers()
        
        for epoch_num in range(
            self.start_epoch + 1, self.config.training.num_epochs + 1
        ):
            print(epoch_num)
            epoch_info = log_utils.StreamingVals()
            self.train_mode()
            self.setup_loaders()
            
            epoch_info = self.train_epoch(epoch_num, epoch_info)

            for scheduler in self.schedulers.values():
                scheduler.step()

            self.eval_mode()
            epoch_info = self.validation_epoch(epoch_num, epoch_info)
            
            self.test_mode()
            epoch_info = self.test_epoch(epoch_num, epoch_info)
            
            self.logger.log_epoch(self.train_iter, epoch_num, epoch_info)

            if not self.config.checkpoint.checkpointing_off:
                checkpoint_dir = os.path.join(
                    self.run_dir, self.config.checkpoint.checkpoint_dir
                )
                with open("epoch_info_detailed.log", "w") as f:
                    for key, value in epoch_info.items():
                        f.write(f"{key}: {type(value)}\n")
                        if hasattr(value, '__dict__'):
                            for attr, val in value.__dict__.items():
                                f.write(f"    {attr}: {val}\n") 
                
                if epoch_num % self.config.checkpoint.save_every == 0:
                    self.chp_logger.save(
                        checkpoint_dir,
                        self.config.exp.name,
                        epoch_num,
                        self.nets,
                        self.optims,
                        self.config.training.device,
                        epoch_info,
                    )
                
                self.logger.update_summary('chp_info', self.chp_logger.history)
