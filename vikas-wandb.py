import argparse
import augmentations as aug
import torch
from torch import nn
import torchvision.datasets as datasets
from pathlib import Path
from VICREG import VICReg
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
#import bitsandbytes as bnb
import wandb
from torchlars import LARS
import gc

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config.augmentation == "normal":
            transforms = aug.TrainTransform()
        elif config.augmentation == "ori":
            transforms = aug.TrainTransformOriGrayScale()
        elif config.augmentation == "oricolor":
            transforms = aug.TrainTransformOriColor()
        elif config.augmentation == "orirandomcolor":
            transforms = aug.TrainTransformOriRandomColor()
        elif config.augmentation == "oricolorjitter":
            transforms = aug.TrainTransformOriColorJitter()
        
        dataset = datasets.ImageFolder('../content/test_donkey_images/', transforms)

        loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle = True
            )

        # num_epochs=1
        learning_rate=config.learning_rate

        model = VICReg(config).to(device)
        if config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        elif config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

        if config.lars:
            optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)
        
        for _ in range(config.epochs):
            running_loss = 0.0
            for step, ((x,y), _) in enumerate(loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad()
                total_loss = model.forward(x,y)
                running_loss += total_loss.item()
                if not torch.isnan(total_loss):
                    total_loss.backward()
                    optimizer.step()
            wandb.log({"batch_loss": running_loss})
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    #train()
    sweep_config = {
        'method': 'random'
    }
    
    parameters_dict = {
        'optimizer': {
            'value': 'sgd'
            },
        'mlp': {
            'values': ['4096-1024','4096-2048','2048-1024','2048-2048']
            },
        'activation': {
            'values': ['lrelu','gelu']
            },
        'lars': {
            'value': True
            },
        'augmentation': {
            'values': ['normal','ori','oricolor','orirandomcolor','oricolorjitter']
            }
        }

    parameters_dict.update({
        'epochs': {
            'value': 3
        },
        'batch_size': {
            'value': 64
        }
    })

    parameters_dict.update({
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.2
        }
    })

    sweep_config['parameters'] = parameters_dict
    sweep_config['program'] = "vikas-wandb.py"

    sweep_id = wandb.sweep(sweep_config, project="pytorch-vikas-v13")
    wandb.agent(sweep_id, train, count=108)



