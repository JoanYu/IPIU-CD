# imports

import os
import datetime
import logging
import random
from utils.parser import get_parser_with_args
from utils.helpers import get_loaders, load_model, get_criterion, initialize_metrics, set_metrics, get_cd_corrects, get_cd_report
from tensorboardX import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')


# init Parser & define args
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

# init logs
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

# device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)

train_loader, val_loader = get_loaders(opt)

logging.info('LOADING Model')
model = load_model(opt, dev)
criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# set some params and values

best_metrics = {i:-1 for i in opt.metrics_test}
logging.info('STARTING training')
total_step = -1
for epoch in range(opt.epochs):
    train_metrics = initialize_metrics(*opt.metrics)
    val_metrics = initialize_metrics(*opt.metrics)
    # Begin Training
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("epoch {} ".format(epoch))
        total_step += 1

        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds = model(batch_img1, batch_img2)

        cd_loss = criterion(cd_preds, labels)
        loss = cd_loss
        loss.backward()
        optimizer.step()
        
        cd_correct = get_cd_corrects(cd_preds, labels, opt.patch_size)
        cd_train_report = get_cd_report(cd_preds, labels)
        train_metrics = set_metrics(train_metrics, 
                                    list(train_metrics.keys()), 
                                    [cd_loss, 
                                     cd_correct, 
                                     cd_train_report[0], 
                                     cd_train_report[1], 
                                     cd_train_report[2], 
                                     scheduler.get_lr()
                                     ]
                                    )

