import argparse
import string
import augmentations as aug
import torch
from torch import nn
import torchvision.datasets as datasets
from pathlib import Path
from VICREG import VICReg
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchlars import LARS
#import bitsandbytes as bnb

def main(args):
    print(args)

    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = aug.TrainTransformOriColorJitter()
    dataset = datasets.ImageFolder(args.data_dir, transforms)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle = True
        )

    # num_epochs=1
    learning_rate=0.15

    model = VICReg(args).to(device)
    if args.loadStoredModel == 0:
        print("Loading Model")
        model.load_state_dict(torch.load(args.data_dir / "model.pth"))
    #Defining the optimizer and lr scheduler:
    #optimizer = bnb.optim.AdamW8bit(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
    optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    #scaler = torch.cuda.amp.GradScaler()
    best_epoch_loss = 1e10
    log_Step=0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, ((x,y), _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            total_loss, invarinace_loss, variance_loss, loss_cov = model.forward(x,y)
            if not torch.isnan(total_loss):
                total_loss.backward()
                optimizer.step()
            if step%50==0:
                print(total_loss.item())
                writer.add_scalar("Total Loss", total_loss.item(),log_Step)
                writer.add_scalar("Invarinace Loss", invarinace_loss.item(),log_Step)
                writer.add_scalar("Variance Loss", variance_loss.item(),log_Step)
                writer.add_scalar("Covariance Loss", loss_cov.item(),log_Step)
                log_Step=log_Step+1
            running_loss += total_loss.item()
        if(running_loss<best_epoch_loss):
            best_epoch_loss = running_loss
            print("[Info] Found New Best Model With Loss: ", best_epoch_loss)
            torch.save(model.state_dict(), args.data_dir / "model.pth")
            torch.save(model.resnet.state_dict(), args.data_dir / "resnet18.pth")
        print('Epoch Loss: {:.4f}'.format(running_loss))

    writer.flush()
    writer.close()
    print("Ending Program")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation Learning for DonkeyCar")
    parser.add_argument("--data-dir", type=Path, required=True, help='Path to the input Images')
    parser.add_argument("--mlp", default="4096-1024",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--loadStoredModel", type=int, default=-1,
                        help='Load Previous model')
    parser.add_argument("--batch-size", type=int, default=90,
                        help='Model Batch Size')
    parser.add_argument("--activation", default="gelu",
                        help='Activation Function')
    args = parser.parse_args()
    main(args)





