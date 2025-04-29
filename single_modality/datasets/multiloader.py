import torch
import torch.distributed as dist
from utils import is_dist_avail_and_initialized
import random
import logging

logger = logging.getLogger(__name__)


class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, loaders):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.

        loaders: Dict, {name: dataloader}
        """
        self.loaders = loaders
        iter_order = [i for i in range(len(self.loaders))]
        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
        self.iter_order = [e for e in iter_order.cpu()]

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """ this iterator will run indefinitely """
        while True:
            try:
                for i in self.iter_order:
                    _iter = self.loaders[i]
                    batch = next(_iter)
                    yield batch
            except:
                iter_order = [i for i in range(len(self.loaders))]
                random.shuffle(iter_order)
                iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)
                if is_dist_avail_and_initialized():
                    # make sure all processes have the same order so that
                    # each step they will have data from the same loader
                    dist.broadcast(iter_order, src=0)
                self.iter_order = [e for e in iter_order.cpu()]
                continue