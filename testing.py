
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
    parser.add_argument('-test_manifest',type=str,default='manifest/test')
    
    parser.add_argument('-num_classes', action="store_true", default=12)
    parser.add_argument('-test_batch_size', action="store_true", default=256)
    
    parser.add_argument('-use_gpu', action="store_true", default=True)
    
    parser.add_argument('-save_dir', type=str, default='save_models')
    parser.add_argument('-num_epochs', action="store_true", default=100)
    parser.add_argument('-save_interval', action="store_true", default=1000)
    parser.add_argument('-log_interval', action="store_true", default=100)
    parser.add_argument('-lr', action="store_true", default=0.001)
    
    args = parser.parse_args()
    return args



def test(model, device, test_loader):
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
    
    return test_acc


  
def main():
    
    args = parse_args()
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
    
    checkpoint = torch.load('/home/krishna/Krishna/Speech/paper_implementations/Keyword-Transformer/trained_models_test/check_point_92_0.9421808304229724')
    model.load_state_dict(checkpoint['model'])
    testset = SpeechGenerator(manifest_file=args.test_manifest,max_len=98)
    test_loader = DataLoader(testset,batch_size=args.test_batch_size,shuffle=False,num_workers=8,pin_memory=True, collate_fn=collate_fun)
    
    
    acc = test(model, device, test_loader)  # evaluate at the end of epoch
    print(f'Test data accuracy {acc*100}')

if __name__ == '__main__':
    main()
    
    
