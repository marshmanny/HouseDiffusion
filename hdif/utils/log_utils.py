import copy
import time
import datetime
import collections
import torch
import torchvision
import numpy as np
import logging
import os
import glob
import heapq
import wandb
import omegaconf
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from PIL import Image

def print_network(net, net_name, x_input, logger, log_complexity):
    input_shape = tuple(x_input.shape)[1:]
    print("input shape:", input_shape)
    print(net)
    if log_complexity:
        macs, params = get_model_complexity_info(
            net,
            input_shape,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        logger.exp_logger.log({f"{net_name}_size": f"{params}"})
        logger.exp_logger.log({f"{net_name}_macs": f"{macs}"})
    print(
        "Number of parameters: {}".format(
            sum(map(lambda p: p.numel(), net.parameters()))
        )
    )


def strf_time_delta(td):
    td_str = ""
    if td.days > 0:
        td_str += f"{td.days} days, " if td.days > 1 else f"{td.days} day, "
    hours = td.seconds // 3600
    if hours > 0:
        td_str += f"{hours}h "
    minutes = (td.seconds // 60) % 60
    if minutes > 0:
        td_str += f"{minutes}m "
    seconds = td.seconds % 60 + td.microseconds * 1e-6
    td_str += f"{seconds:.1f}s"
    return td_str


class LoggingManager:
    def __init__(self, exp_logger, console_logger):
        self.exp_logger = exp_logger
        self.console_logger = console_logger

    def log_scalar(self, global_iter_num, epoch_num, iter_info, event=None, **kwargs):
        self.exp_logger.log_scalar(global_iter_num, epoch_num, iter_info, **kwargs)
    
    def log_images(self, table_name, images, iter=1, epoch_num=1):
        self.exp_logger.log_images(table_name, images, iter, epoch_num)

    def log_epoch(self, global_iter_num, epoch_num, epoch_info):
        scalar_dict = {}
        
        for k, v in epoch_info.items():
            val_type = v.val_type
            val = v.val
        
            if val_type == "num":
                scalar_dict[k] = val
            elif val_type == "img":
                self.exp_logger.log_images(k, val, iter=global_iter_num, epoch_num=epoch_num)
            else:
                assert False, f"[error] Cant log type: {val_type}"
        
        self.exp_logger.log_scalar(global_iter_num, epoch_num, scalar_dict)
    
    def log_info(self, output_info):
        self.console_logger.logger.info(output_info)
    
    def update_summary(self, key, value):
        self.exp_logger.update_summary(key, value)
    
    def update_dataset_info(self, name, stage, length):
        self.exp_logger.update_dataset_info(name, stage, length)


class WandbLogger:
    def __init__(self, omega_config=None, save_code=True, **kwargs):
        wandb.login(key=omega_config.personal.wandb_key, relogin=True)
        wandb.init(**kwargs)
        self.run_dir = wandb.run.dir
        root_dir = os.path.abspath(".")
        full_path = lambda x: os.path.abspath(os.path.join(root_dir, x))
        
        code = wandb.Artifact("project-source", type="code")
        if save_code:
            for path in glob.glob("**/*.py", recursive=True, root_dir=root_dir):
                if not path.startswith("wandb"):
                    if os.path.basename(path) != path:
                        code.add_dir(
                            os.path.dirname(full_path(path)), name=os.path.dirname(path)
                        )
                    else:
                        code.add_file(full_path(path), name=os.path.basename(path))
        
        if omega_config is not None:
            omegaconf.OmegaConf.save(config=omega_config, f=os.path.join(self.run_dir, 'launch_config.yml'))
            code.add_file(os.path.join(self.run_dir, 'launch_config.yml'), name='launch_config.yml')
        wandb.run.log_artifact(code)

    def log(self, info):
        wandb.log(info)

    def log_images(self, name, imgs, iter, epoch_num):
        wandb.log({name: [wandb.Image(img) for img in imgs]}, step=epoch_num)
    
    def log_scalar(self, iter_num, epoch_num, iter_info):
        wandb.log({**iter_info, "iter": iter_num, "epoch": epoch_num}, step=epoch_num, commit=False)
        
    def update_summary(self, key, value):
        wandb.run.summary[key] = value


class ConsoleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @staticmethod
    def format_info(info):
        if not info:
            return str(info)
        log_groups = collections.defaultdict(dict)
        for k, v in info.to_dict().items():
            prefix, suffix = k.split("/", 1)
            log_groups[prefix][suffix] = (
                f"{v:.3f}" if isinstance(v, float) else str(v)
            )
        formatted_info = ""
        max_group_size = len(max(log_groups, key=len)) + 2
        max_k_size = (
            max([len(max(g, key=len)) for g in log_groups.values()]) + 1
        )
        max_v_size = (
            max([len(max(g.values(), key=len)) for g in log_groups.values()])
            + 1
        )
        for group, group_info in log_groups.items():
            group_str = [
                f"{k:<{max_k_size}}={v:>{max_v_size}}"
                for k, v in group_info.items()
            ]
            max_g_size = len(max(group_str, key=len)) + 2
            group_str = "".join([f"{g:>{max_g_size}}" for g in group_str])
            formatted_info += f"\n{group + ':':<{max_group_size}}{group_str}"
        return formatted_info

    def log_iter(
        self, epoch_num, iter_num, num_iters, iter_info, event="epoch"
    ):
        output_info = (
            f"{event.upper()} {epoch_num}, ITER {iter_num}/{num_iters}:"
        )
        output_info += self.format_info(iter_info)
        self.logger.info(output_info)

    def log_epoch(self, epoch_num, epoch_info):
        output_info = f"EPOCH {epoch_num}:"
        output_info += self.format_info(epoch_info)
        self.logger.info(output_info)


class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info:
            self.info[f"duration/{self.log_event}"] = self.duration


class TimeLog:
    def __init__(self, logger, total_num, event):
        self.logger = logger
        self.total_num = total_num
        self.event = event.upper()
        self.start = time.time()

    def now(self, current_num):
        elapsed = time.time() - self.start
        left = self.total_num * elapsed / (current_num + 1) - elapsed
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        left = strf_time_delta(datetime.timedelta(seconds=left))
        self.logger.log_info(
            f"TIME ELAPSED SINCE {self.event} START: {elapsed}"
        )
        self.logger.log_info(f"TIME LEFT UNTIL {self.event} END: {left}")

    def end(self):
        elapsed = time.time() - self.start
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        self.logger.log_info(
            f"TIME ELAPSED SINCE {self.event} START: {elapsed}"
        )
        self.logger.log_info(f"{self.event} ENDS")

class CheckpointLogger:
    def __init__(self, config):
        self.config = config
        self.history = []


    def save(self, checkpoint_dir, exp_name, epoch_num, nets, optims, device, epoch_info={}):
        if self.config.metric not in epoch_info:
            raise KeyError(f"{self.config.metric} отсутствует в epoch_info. Доступные ключи: {epoch_info.keys()}")
        cur_metric = epoch_info[self.config.metric].val if hasattr(epoch_info[self.config.metric], 'val') else None
        if cur_metric is None:
            raise ValueError(f"Metric {self.config.metric} не содержит атрибут .val")

        
        if len(self.history) < self.config.max_amount:
            self.history.append({"metric": cur_metric, "epoch": epoch_num})
            
            save_checkpoint(checkpoint_dir,
                            exp_name,
                            epoch_num,
                            nets,
                            optims,
                            device,
                            save_full=False)
        else:
            found_worther = False
            for chp in self.history:
                if cur_metric < chp['metric']:
                    found_worther = True
            
            if found_worther:
                remove_checkpoint(checkpoint_dir,
                                self.history[0]["epoch"],
                                nets,
                                optims,
                                True)
                self.history = [{"metric": cur_metric, "epoch": epoch_num}] + self.history[1:]
                
                save_checkpoint(checkpoint_dir,
                            exp_name,
                            epoch_num,
                            nets,
                            optims,
                            device,
                            save_full=False)
        self.history = list(sorted(self.history, key=lambda x: x['metric']))[::-1]
        with open(os.path.join(checkpoint_dir, '..', 'chp_history.txt'), 'w') as f:
            f.write(str([{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in x.items()} for x in self.history]))

def check_is_num_or_img(val):
    val_type = "num"
    if isinstance(val, torch.Tensor):
        if len(val.shape) > 1:
            val_type = "img"
        else:
            val_type = "num"
    elif isinstance(val, Image.Image):
        val_type = "img"
    elif isinstance(val, str):
        val_type = "img"
    return val_type

def check_type(val):
    is_list = isinstance(val, list)
    if is_list:
        val = val[0]
    val_type_check = check_is_num_or_img(val)
    if val_type_check is None:
        assert False, "Cant assingn type"
    return val_type_check

class _StreamingNumVals:
    def __init__(self, val=None, counts=None):
        # need for wandb logging
        self.val_type = "num"
        
        if val is None:
            self.val = 0.0
            self.counts = 0
        else:
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1
            
            is_list = isinstance(val, list)
            if is_list:
                self.counts = len(val)
                val = torch.mean(torch.Tensor(val))

            if isinstance(val, torch.Tensor):
                val = val.item()
            self.val = val
            
    def update(self, val, counts=1):
        if not isinstance(val, _StreamingNumVals):
            assert False
        
        val, counts = val.val, val.counts * counts
        if counts == 0:
            return
        total = self.counts + counts
        
        self.val = (self.counts * self.val) / total + (counts * val) / total
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.val, self.counts)
        if isinstance(other, _StreamingNumVals):
            if other.counts == 0:
                return new
            else:
                new.update(other.val, other.counts)
        else:
            new.update(other)
        return new

class _StreamingListVals:
    def __init__(self, val=None, counts=None):
        # need for wandb logging
        self.val_type = "img"
        
        if val is None:
            self.val = []
            self.counts = 0
        else:
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1
            
            is_list = isinstance(val, list)
            if is_list:
                self.counts = len(val)
            else:
                val = [val]
            
            self.val = val
            
    def update(self, val, counts=1):
        if not isinstance(val, _StreamingListVals):
            assert False
        
        val, counts = val.val, val.counts * counts
        if counts == 0:
            return
        total = self.counts + counts
        self.val.extend(copy.deepcopy(val))
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.val, self.counts)
        if isinstance(other, _StreamingListVals):
            if other.counts == 0:
                return new
            else:
                new.update(other.val, other.counts)
        else:
            new.update(other)
        return new

class _StreamingMixVals:
    def __new__(cls, val_type):
        # случайно выбранный `other`
        instance = STREAMING_MAP[val_type]
        return instance

STREAMING_MAP = {
    "num": _StreamingNumVals,
    "img": _StreamingListVals,
}

class StreamingVals(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMixVals)
        self._key_to_val_type = {}

    def __setitem__(self, key, value):
        if key not in self.keys():
            self._key_to_val_type[key] = check_type(value)
        
        if isinstance(value, _StreamingMixVals):
            final_val = value
        else:
            final_val = _StreamingMixVals(self._key_to_val_type[key])(value)
        
        if key in self.keys():
            self[key].update(final_val)
        else:
            super().__setitem__(key, final_val)

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            if not hasattr(v, 'val_type'):
                v = _StreamingMixVals(check_type(v))(v)
            
            if k in self.keys():
                self[k].update(v)
            else:
                super().__setitem__(k, _StreamingMixVals(v.val_type)())
                self[k].update(v)
        
    def to_dict(self, prefix=""):
        return dict((prefix + k, v.val) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])

def save_checkpoint(
    checkpoint_dir, exp_name, epoch_num, nets, optims, device, save_full=False
):
    for name, net in nets.items():
        net = net.cpu()
        torch.save(
            net.state_dict(),
            os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth"),
        )
        net.to(device)
    if save_full:
        for name, optim in optims.items():
            if optim is not None:
                torch.save(
                    optim.state_dict(),
                    os.path.join(checkpoint_dir, f"{name}_{epoch_num:04d}.pth"),
                )
    # save exp_name in exp dir (parent of the checkpoints) to make restoring
    # from checkpoint for the same exp_name possible
    with open(os.path.join(checkpoint_dir, '..', 'exp_name.txt'), 'w') as f:
        f.write(exp_name)


def restore_checkpoint_from_dir(checkpoint_dir, epoch_num, nets, device):
    if epoch_num == -1:
        epoch_nums = [os.path.basename(x).split('_')[-1].split('.')[0].lstrip('0') for x in os.listdir(checkpoint_dir) if "pth" in x]
        epoch_num = list(sorted(epoch_nums, key=lambda x: int(x)))[-1]
        epoch_num = int(epoch_num)
    
    for name, net in nets.items():
        net.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_dir, f"{name}_{epoch_num:04d}.pth"
                ),
                map_location=device
            )
        )

def restore_checkpoint_from_path(checkpoint_path, epoch_num,
                                nets, optims, device):
    for name, net in nets.items():
        net.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_path, f"{name}_{epoch_num:04d}.pth"
                ),
                map_location=device
            )
        )
    for name, optim in optims.items():
        optim.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_path, f"{name}_{epoch_num:04d}.pth"
                ),
                map_location=device
            )
        )

def restore_checkpoint(run_dir, exp_name, nets, optims, device,
                       logger=logging.getLogger('root')):
    """
    Load latest checkpoint for the experiment with the same name.
    If there are multiple latest checkpoints, load any one of them.
    Return the epoch for the latest checkpoint that was loaded,
    or -1 if no valid checkpoint was found.
    """
    # find wandb experiment run with the same name
    checkpoint_dir_candidates = glob.glob(
        os.path.join(run_dir, "..", "..", "*", "files")
    )
    checkpoint_dirs = list()
    for dn in checkpoint_dir_candidates:
        exp_name_fn = os.path.join(dn, "exp_name.txt")
        if not os.path.exists(exp_name_fn):
            continue
        with open(exp_name_fn, "r") as f:
            candidate_exp_name = f.read()
        if candidate_exp_name == exp_name:
            checkpoint_dir = os.path.realpath(os.path.join(dn, "checkpoints"))
            checkpoint_dirs.append(checkpoint_dir)
    if len(checkpoint_dirs) == 0:
        return -1
    logger.info(
        "Restore checkpoint: directories found: " +
        ", ".join(checkpoint_dirs)
    )

    # make a sorted list of epochs to try loading
    epochs = list()
    for checkpoint_dir in checkpoint_dirs:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*_*.pth"))
        epochs.extend([
            int(os.path.basename(c).split("_")[-1][:-4])
            for c in checkpoints
        ])
    epochs = list(set(epochs))
    epochs = list(reversed(sorted(epochs)))
    if len(epochs) == 0:
        logger.warning(
            "Restore checkpoint: no suitable checkpoints were found"
        )
        return -1

    # we want to restore initial initialization if something goes wrong
    backup_nets = dict()
    for name, net in nets.items():
        backup_nets[name] = copy.deepcopy(net.state_dict())
    backup_optims = dict()
    for name, optim in optims.items():
        backup_optims[name] = copy.deepcopy(optim.state_dict())
    for epoch_num in epochs:
        for checkpoint_dir in checkpoint_dirs:
            try:
                restore_checkpoint_from_dir(checkpoint_dir, epoch_num, nets,
                                            optims, device)
                # if we are here, we successfully loaded the checkpoint
                logger.info(
                    f"Restore checkpoint: loaded checkpoint for epoch "
                    f"{epoch_num:04d} from {checkpoint_dir}"
                )
                return epoch_num
            except Exception as e:
                # just suppress because there might be a lot of such cases
                pass
        logger.warning(
            "Restore checkpoint: failed to load checkpoint for epoch " +
            f"{epoch_num:04d}"
        )

    # here we failed to load any checkpoint; hence restoring from backups
    for name, net in nets.items():
        net.load_state_dict(backup_nets[name])
    for name, optim in optims.items():
        optim.load_state_dict(backup_optims[name])
    logger.warning(
        "Restore checkpoint: failed to load any checkpoint, "
        "using standard initialization"
    )
    return -1


def remove_checkpoint(checkpoint_dir, epoch_num, nets, optims, remove_full):
    for name in list(nets.keys()) + list(optims.keys()):
        if name == "gen" and not remove_full:
            continue
        fn = os.path.join(
            checkpoint_dir, f"{name}_{epoch_num:04d}.pth"
        )
        if os.path.exists(fn):
            os.remove(fn)

class ModelLogging:
    def __init__(self, logger, conditional=False):
        self.tough_samples = collections.defaultdict(list)
        self.logger = logger
        self.conditional = conditional

    def log_disc(
        self,
        iter_info,
        logits_real,
        logits_fake,
        x_real,
        x_fake,
        log_event,
        num_tough_samples=5,
    ):
        x_real = x_real[0] if self.conditional else x_real
        x_fake = x_fake[0] if self.conditional else x_fake
        iter_info.update(
            {
                f"disc_log_{log_event}/real_mean": logits_real.mean(),
                f"disc_log_{log_event}/real_med": logits_real.median(),
                f"disc_log_{log_event}/acc_real": trainer_utils.accuracy(
                    real_logits=logits_real
                ),
                f"disc_log_{log_event}/fake_mean": logits_fake.mean(),
                f"disc_log_{log_event}/fake_med": logits_fake.median(),
                f"disc_log_{log_event}/acc_fake": trainer_utils.accuracy(
                    fake_logits=logits_fake
                ),
            }
        )
        real_key = f"disc_tough_samples/{log_event}/real"
        self.tough_samples[real_key] = heapq.nsmallest(
            num_tough_samples,
            list(
                zip(logits_real.view(logits_real.size(0), -1).mean(1), x_real)
            )
            + self.tough_samples[real_key],
            key=lambda t: t[0],
        )
        fake_key = f"disc_tough_samples/{log_event}/fake"
        self.tough_samples[fake_key] = heapq.nlargest(
            num_tough_samples,
            list(
                zip(logits_fake.view(logits_fake.size(0), -1).mean(1), x_fake)
            )
            + self.tough_samples[fake_key],
            key=lambda t: t[0],
        )

    def log_tough_samples(self, epoch_num):
        for name, samples in self.tough_samples.items():
            for i, (logit, x) in enumerate(samples):
                caption = f"epoch = {epoch_num}, logit = {logit:.3f}"
                wandb.log({f"{name}_{i}": wandb.Image(x, caption=caption)})
