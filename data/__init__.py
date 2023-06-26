import torch.utils.data
from .base_dataset import collate_fn
from .base_dataset import collate_fn_2
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset
def train_val_dataset(dataset, val_split=0.5):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, val_idx)
    return datasets

def create_dataset(opt):
    """loads dataset class"""
    from .classification_data import ClassificationData
    dataset = ClassificationData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = create_dataset(opt)
        self.dataloaders = {}
        if self.opt.dataset_name == 'SHREC_Split10':
            datasets = train_val_dataset(self.dataset, val_split=0.5)
            # self.dataloaders = {}
            self.dataloaders['train'] = torch.utils.data.DataLoader(
                                            datasets['train'],
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_threads,
                                            collate_fn=collate_fn, drop_last=True)
            self.dataloaders['test'] = torch.utils.data.DataLoader(
                                            datasets['test'],
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.num_threads,
                                            collate_fn=collate_fn, drop_last=False)
            return

        if opt.save_features:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_threads,
                collate_fn=collate_fn_2, drop_last=True)

        if opt.phase == 'train' and not opt.save_features:
            self.dataloaders['train'] = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_threads,
                collate_fn=collate_fn, drop_last=True)

        if (opt.phase == 'test'):
            self.dataloaders['test'] = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_threads,
                collate_fn=collate_fn, drop_last=False)


    def __len__(self):
        return len(self.dataset)
