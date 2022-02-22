import os
import sys
import time
import subprocess
import argparse
import warnings

import torch
import torch.distributed as dist
from torch.autograd import Variable
from config import _C as config

warnings.filterwarnings('ignore')

def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= num_gpus
    return rt

def init_distributed(rank, num_gpus, group_name, dist_backend, dist_url):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(dist_backend, init_method=dist_url,
                            world_size=num_gpus, rank=rank,
                            group_name=group_name)

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def apply_gradient_allreduce(module):
    """
    Modifies existing model to do gradient allreduce, but doesn't change class
    so you don't need "module"
    """
    if not hasattr(dist, '_backend'):
        module.warn_on_half = True
    else:
        module.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        if(module.needs_reduction):
            module.needs_reduction = False
            buckets = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = type(param.data)
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
            if module.warn_on_half:
                if torch.cuda.HalfTensor in buckets:
                    print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                          " It is recommended to use the NCCL backend in this case. This currently requires" +
                          "PyTorch built from top of tree master.")
                    module.warn_on_half = False

            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):
        def allreduce_hook(*unused):
            Variable._execution_engine.queue_callback(allreduce_params)
        if param.requires_grad:
            param.register_hook(allreduce_hook)
            dir(param)

    def set_needs_reduction(self, input, output):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def main(config_file, opts):
    num_gpus = torch.cuda.device_count()
    os.makedirs(config.save_dir, exist_ok=True)
    workers = []
    for i in range(num_gpus):
        args_list = ['train.py']
        if config_file != '':
            args_list.extend(["--config_file", config_file])
        args_list.extend(["--rank", str(i)])
        args_list.extend(["--group_name", "group_{}".format(time.strftime("%Y_%m_%d-%H%M%S"))])
        args_list += opts if len(opts) > 0 else []
        stdout = None if i == 0 else open(os.path.join(config.save_dir, "GPU_{}.log".format(i)), "a+")
        p = subprocess.Popen([str(sys.executable)]+args_list, stdout=stdout)
        workers.append(p)

    for p in workers:
        p.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='',
                        help='file for configuration')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    config.merge_from_list(args.opts)
    config.freeze()

    main(args.config_file, args.opts)
