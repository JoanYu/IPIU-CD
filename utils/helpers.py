import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader)
from utils.loss import jaccard_loss, dice_loss, hybrid_loss
from models.siam_nestedunet_ecam import SNUNet_ECAM
from models.dasnet import SiameseNet
logging.basicConfig(level=logging.INFO)

def get_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def load_model(opt, device):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    device_ids = list(range(opt.num_gpus))
    if opt.model == 'siamunet++':
        logging.info('Model: SNUNet_ECAM')
        model = SNUNet_ECAM(opt.num_channel, opt.label_channel)
    elif opt.model == 'dasnet':
        logging.info('Model: DASNet')
        model = SiameseNet(opt.num_channel, opt.label_channel)
    model = model.to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)
    return model

def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion

def initialize_metrics(*args):
    metrics = {}
    for i in args:
        metrics[i] = []
    return metrics
    
def set_metrics(metrics, metrics_keys, metrics_values):
    for i in metrics_keys:
        metrics[i].append(metrics_values[i])
    return metrics

def get_cd_corrects(cd_preds, labels, patch_size):
    return (100 *
            (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() //
            (labels.size()[0] * (patch_size**2)))

def get_cd_report(cd_preds, labels):
    return prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)