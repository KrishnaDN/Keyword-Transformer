#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:06:06 2021

@author: krishna
"""

import os
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import apex.amp as amp
import argparse
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.transformer import ViT
from dataloader import SpeechGenerator,collate_fun
from model.label_smoothing import LabelSmoothingLoss
from sklearn.metrics import accuracy_score
besteer=99
########## Argument parser
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-train_manifest',type=str,default='manifest/train')
    parser.add_argument('-dev_manifest',type=str,default='manifest/valid')
    
    parser.add_argument('-num_classes', action="store_true", default=12)
    parser.add_argument('-train_batch_size', action="store_true", default=512)
    parser.add_argument('-dev_batch_size', action="store_true", default=512)
    
    parser.add_argument('-use_gpu', action="store_true", default=True)
    
    parser.add_argument('-save_dir', type=str, default='save_models')
    parser.add_argument('-num_epochs', action="store_true", default=100)
    parser.add_argument('-save_interval', action="store_true", default=1000)
    parser.add_argument('-log_interval', action="store_true", default=100)
    parser.add_argument('-lr', action="store_true", default=0.001)
    
    args = parser.parse_args()
    return args



def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    iteration = 0
    model.train()  # set training mode
    start = time()
    gt_labels = list()
    pred_labels =list()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        correct=0.0
        feats = torch.stack(data)
        target = torch.stack(target).squeeze(1)
        data, target = feats.to(device), target.to(device)
        output = model(data)
        
        criterion = nn.CrossEntropyLoss().to(device)
        label_smoothing = LabelSmoothingLoss(size=12, padding_idx=0, smoothing=0.1)
        
        loss = criterion(output, target) + 0.0*label_smoothing(output, target)
        train_loss += loss
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / len(target)
        if iteration % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} Acc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx / len(train_loader), loss.item(), acc
            ))
            # log using SummaryWriter
        for lab in target.detach().cpu().numpy():
            gt_labels.append(lab)
        
        for lab in pred.detach().cpu().numpy():
            pred_labels.append(lab)
    
        iteration += 1
    
    end = time()
    train_loss /= len(train_loader.dataset)
    print(f'Train loss {train_loss} after Epoch {epoch}')
    print(f'Total accuracy {accuracy_score(gt_labels, pred_labels)} after Epoch {epoch}')

def test(model, device, test_loader, ep):
    model.eval()
    test_loss = 0
    correct = 0
    print('length of devset: ' + str(len(test_loader.dataset)))
    start = time()
    correct = 0
    gt_labels = list()
    pred_labels =list()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            
            feats = torch.stack(data)
            target = torch.stack(target).squeeze(1)
            data, target = feats.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss().to(device)
            label_smoothing = LabelSmoothingLoss(size=12, padding_idx=0, smoothing=0.1)
        
            loss = criterion(output, target) + 0.0*label_smoothing(output, target)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for lab in target.detach().cpu().numpy():
                gt_labels.append(lab)
        
            for lab in pred.detach().cpu().numpy():
                pred_labels.append(lab)
    
    end = time()
    test_loss /= len(test_loader.dataset)
    

    test_acc = accuracy_score(gt_labels, pred_labels)
    print(f'Test loss {test_loss} after Epoch {ep}')
    print(f'Test accuracy {test_acc} after Epoch {ep}')
    
    
    return test_acc



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



  
def main():
    
    args = parse_args()
    setup_seed(2018)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use", device)
    model = ViT(
        image_size = (98,40),
        patch_size = (2,20),
        num_classes = 12,
        dim = 64,
        depth = 12,
        heads = 4,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
        ).to(device)
    
        
    trainset = SpeechGenerator(manifest_file=args.train_manifest,max_len=98)
    train_loader = DataLoader(trainset,batch_size=args.train_batch_size,shuffle=True,num_workers=8,pin_memory=True, collate_fn=collate_fun)
    
    
    devset = SpeechGenerator(manifest_file=args.dev_manifest,max_len=98)
    dev_loader = DataLoader(devset,batch_size=args.dev_batch_size,shuffle=False,num_workers=8,pin_memory=True, collate_fn=collate_fun)
    
    
    optimizer = optim.AdamW(model.parameters(),lr=args.lr, weight_decay = 0.1)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)
    
    for epoch in range(args.num_epochs):
        
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        
        acc = test(model, device, dev_loader, epoch)  # evaluate at the end of epoch
        scheduler.step(acc)
        model_save_path = os.path.join('trained_models_test', 'check_point_'+str(epoch)+'_'+str(acc))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
    
if __name__ == '__main__':
    main()
    
    
