import os
import time
import copy
import argparse

import numpy as numpy
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from data_load import *
from model import UNet


def train(config):

    # 0. config
    batch_size = config.batch_size
    num_worker = config.num_worker
    n_epoch = config.epoch
    seed = config.data_load_seed
    path = config.data_path
    wandb.init(project='UNet_reproduce', entity='heystranger')  #wandb

    # 1. data load
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    dataset_train = Dataset(
        data_dir=os.path.join(path, 'train'),
        transform=transform_train,
        seed=seed)
    dataset_val = Dataset(
        data_dir=os.path.join(path, 'val'),
        transform=transform_val,
        seed=seed)

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker
    )
    loader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker
    )

    # 2. variable setting
    n_traindata = len(dataset_train)
    n_valdata = len(dataset_val)
    n_batch_train = np.ceil(n_traindata/batch_size)
    n_batch_val = np.ceil(n_valdata/batch_size)


    # 3. network setting
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)


    # 4. training
    start_time = time.time()

    best_model_weight = copy.deepcopy(net.state_dict())
    best_loss = 100 #initial loss

    for epoch in range(1, n_epoch+1):
        net.train()
        loss_arr = []

        # train
        for batch, data in enumerate(loader_train, 1):
            data['label'] = data['label']*0.5 + 0.5  # label denormalization
            images, labels = data['input'], data['label']
            images, labels = data['input'].to(device), data['label'].to(device)

            output = net(images)

            optim.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optim.step()
            loss_arr += [loss.item()]
            train_loss = np.mean(loss_arr)

            print('Train:  Epoch %04d/%04d  |  Batch %04d/%04d  | Loss %.4f' %
                  (epoch, n_epoch, batch, n_batch_train, train_loss))

        print('##########################################')
        print('Train: Epoch: %04d  | Epoch Loss %.4f' % (epoch, train_loss))
        print('##########################################')
        wandb.log({'Epoch': epoch, 'loss': train_loss})

        # valid
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, (images, labels) in enumerate(loader_val, 1):
                data['label'] = data['label'] * 0.5 + 0.5  # label denormalization
                images, labels = data['input'], data['label']
                images, labels = data['input'].to(device), data['label'].to(device)

                output = net(images)

                loss = criterion(output, labels)
                loss_arr += [loss.item()]
                val_loss = np.mean(loss_arr)

                print('Valid:  Epoch %04d/%04d  |  Batch %04d/%04d  | Loss %.4f' %
                      (epoch, n_epoch, batch, n_batch_val, val_loss))

            epoch_loss = val_loss

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weight = copy.deepcopy(net.state_dict())
        print()


    # 5. saving model
    net.load_state_dict(best_model_weight)
    torch.save(net.state_dict(), './model_log/model_weights.pth')

    # 6. print total training time & best val loss
    total_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best val loss: {:4f}'.format(best_loss))


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--data_load_seed', type=int, default=10)
    parser.add_argument('--num_worker', type=str, default=2)
    parser.add_argument('--epoch', type=int, default=30)

    config = parser.parse_args()
    #print(config)

    train(config)