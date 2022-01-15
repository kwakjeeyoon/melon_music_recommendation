import os
import random
import configparser
import argparse
import yaml
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AudioDataset
from model import autoencoder, ConvAutoencoder
from transform import get_transform
from loss import create_criterion
from optim_sche import get_opt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class Trainer:
    
    def __init__(self, data_path, save_path, seed, wandb):

        self.data_path = data_path
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = seed
        self.wandb = wandb

    def train(self, config):

        print('Start training ...')

        print("pytorch version: {}".format(torch.__version__))
        print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))

        # seed
        seed_everything(self.seed)

        # transform
        transform_train = get_transform()

        # dataset
        train_dataset = AudioDataset(self.data_path, transform = transform_train)

        # dataloader
        dataloader = DataLoader(train_dataset, **config['dataloader'])
        # print(next(iter(dataloader)))
        
        # model
        model_module = getattr(import_module("model"), config['model'])
        model = model_module(batch_size=config['dataloader']['batch_size']) # autoencoder만 batch_size 지정
        model = model.to(self.device)

        if self.wandb == True:
            wandb.watch(model)

        # criterion
        criterion = create_criterion(config['criterion'])

        # optimizer
        optimizer = get_opt(config['optimizer'], model)

        num_epoch = config['train']['max_epoch']
        step = 0

        for i in range(num_epoch):
            for j, audio in enumerate(tqdm(dataloader)):
                x = audio.to(self.device)
                # input = x.view(config['dataloader']['batch_size'], -1).float() # autoencoder
                optimizer.zero_grad()
                output = model.forward(x)
                # output = output.view(config['dataloader']['batch_size'], 1, 48, 1876).float() # autoencoder
                # break
                loss = criterion(output, x.float())
                loss.backward()
                optimizer.step()
                if self.wandb == True:
                    wandb_log = {}
                    wandb_log["Train/loss"] = round(loss.item(), 4)
                    wandb.log(wandb_log, step)
                step += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_train', type=str, help='path of train configuration yaml file')
    args = parser.parse_args()

    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    # wandb init
    if config_train['wandb'] == True:
        wandb.init(
            entity=config_train['entity'],
             project=config_train['project']
        )
        wandb.run.name = config_train['name']
        wandb.config.update(args)

    trainer = Trainer(**config_train['trainer'], wandb = config_train['wandb'])

    model = trainer.train(config_train['default'])

    print('Training Complete!')