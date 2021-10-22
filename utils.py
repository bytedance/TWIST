# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import math
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", fn=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.fn = fn

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    msg = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                    print(msg)
                    if self.fn and is_main_process():
                        with open(self.fn, 'a') as f:
                            f.write(msg+'\n')
                else:
                    msg = log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                    print(msg)
                    if self.fn and is_main_process():
                        with open(self.fn, 'a') as f:
                            f.write(msg+'\n')
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def adjust_learning_rate(args, optimizer, loader, step):
    warmup_steps = args.warmup_epochs * len(loader)
    mme_steps = args.mme_epochs * len(loader)
    sl_warmupsteps = args.sl_warmup_epochs * len(loader)
    max_steps = args.epochs * len(loader)
    base_lr = 1.0
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
        optimizer.param_groups[0]['lr'] = lr * args.lr
        optimizer.param_groups[1]['lr'] = lr * args.lr / args.lr_wbr
    elif step < mme_steps:
        step -= warmup_steps
        cosann_mme_steps = mme_steps - warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / cosann_mme_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]['lr'] = lr * args.lr
        optimizer.param_groups[1]['lr'] = lr * args.lr / args.lr_wbr
    elif step < (mme_steps + sl_warmupsteps):
        step -= mme_steps
        lr = base_lr * step / sl_warmupsteps
        optimizer.param_groups[0]['lr'] = lr * args.lr_sl
        optimizer.param_groups[1]['lr'] = lr * args.lr_sl / args.lr_wbr
    else:
        step -= (mme_steps + sl_warmupsteps)
        max_steps -= (mme_steps + sl_warmupsteps)
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]['lr'] = lr * args.lr_sl
        optimizer.param_groups[1]['lr'] = lr * args.lr_sl / args.lr_wbr

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

class MeanPerClassAccuracy(object):
    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.total = np.zeros(n_cls)
        self.correct = np.zeros(n_cls)

    def add(self, pred, target):
        true_judge = (pred == target)
        for t in target:
            self.total[t] += 1
        for i, t in enumerate(pred):
            if true_judge[i]:
                self.correct[t] += 1
		
    def get(self):
        pca = np.zeros(self.n_cls)
        for i in range(self.total.shape[0]):
            if self.total[i] != 0:
                a = self.correct[i] / self.total[i]
                pca[i] = a
        return pca.mean()


def eleven_point_map(scr, gts):
    """
    Eleven points mAP for evaluating VOC multi-label classification
    scr: probability of binary classification, shape of N x C
    gts: ground truth labels, shape of N x C
    """
    scr = scr.cpu().numpy()
    gts = gts.cpu().numpy()
    #ind = np.argsort(scr, axis=0)[::-1]
    #scr = np.take_along_axis(scr, ind, axis=0)
    #gts = np.take_along_axis(gts, ind, axis=0)
    mean_ap = 0
    for i in range(20):
        current_gts = gts[:,i][gts[:,i]<=1]
        current_scr = scr[:,i][gts[:,i]<=1]
        ind = np.argsort(current_scr, axis=0)[::-1]
        current_gts = current_gts[ind]
        current_scr = current_scr[ind]
        cumgts = np.cumsum(current_gts)
        recall = cumgts/float(current_gts.sum())
        precision = np.array([cumgts[i]/(i+1) for i in range(len(current_gts))])

        milestones = np.linspace(0,1,11)
        ap = 0
        for m in milestones:
            p = (precision[recall>=m]).max()
            ap += p
        ap /= 11.
        mean_ap += ap
    mean_ap /= 20.
    return mean_ap


class L2Norm(torch.nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1)

class ClassHead(nn.Module):
    def __init__(self, args, feature_dim=2048):
        super(ClassHead, self).__init__()
        batchnorm = nn.SyncBatchNorm

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 2048, bias=True),
            batchnorm(2048),
            nn.ReLU() if args.act == 'relu' else nn.GELU(),
            nn.Dropout(p=args.drop),

            nn.Linear(2048, 2048, bias=True),
            batchnorm(2048),
            nn.ReLU() if args.act == 'relu' else nn.GELU(),
            nn.Dropout(p=args.drop),
        )
        self.cls_heads = nn.ModuleList()
        for i in range(args.num_heads):
            self.cls_heads.append(nn.Sequential(
                nn.Linear(2048, args.dim, bias=True),
            ))
        if args.backbone.startswith('vit') or args.proj_trunc_init:
            print('using vit initialization')
            self.apply(self._vit_init_weights)

    def _vit_init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projection_head(x)
        outs = []
        for head in self.cls_heads:
            outs.append(head(x))
        return outs



