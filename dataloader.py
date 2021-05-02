#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:06:14 2021

@author: krishna
"""

import numpy as np
import random
from PIL import Image
from PIL.Image import BICUBIC
import numpy as np
import torch
import torch.nn.functional as F

def _spec_augmentation(x,
                       warp_for_time=False,
                       num_t_mask=2,
                       num_f_mask=2,
                       max_t=20,
                       max_f=5,
                       max_w=40):
    """ Deep copy x and do spec augmentation then return it

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((max_freq,
                                                   max_frames - warped),
                                                   BICUBIC)
        y = np.concatenate((left, right), 0)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y
    


def _spec_substitute(x, max_t=10, num_t_sub=3):
    """ Deep copy x and do spec substitute then return it

    Args:
        x: input feature, T * F 2D
        max_t: max width of time substitute
        num_t_sub: number of time substitute to apply

    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    for i in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = y[start - pos:end - pos, :]
    return y



class SpeechGenerator():
    def __init__(self, manifest_file, max_len=98):
        self.read_links = [line.rstrip('\n') for line in open(manifest_file)]
        self.max_len = max_len
    
    def _get_features(self, npy_path):
        datum = np.load(npy_path, allow_pickle=True).item()
        features = datum['feats']
        labels = datum['label']
        return features, labels
        
    def __len__(self):
        return len(self.read_links)
        
    def __getitem__(self, idx):
        npy_link = self.read_links[idx]
        features, label = self._get_features(npy_link)
        
        if features.shape[0]>self.max_len:
            tensor_feat = torch.Tensor(features[:self.max_len,:])  
        else:
            tensor_feat = F.pad(torch.Tensor(features), (0, 0, 0, self.max_len - features.shape[0]))
        
        return tensor_feat.unsqueeze(0), torch.LongTensor([int(label)])
        

def collate_fun(batch):
    features = []
    labels = []
    for item in batch:
        feats = item[0]
        label = item[1]
        features.append(feats)
        labels.append(label)
        #features.append(torch.Tensor(_spec_augmentation(feats)))
        #labels.append(label)
        #features.append(torch.Tensor(_spec_substitute(feats)))
        #labels.append(label)
    
    return features, labels






