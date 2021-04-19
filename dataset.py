#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:24:15 2021

@author: krishna
"""

import os
import numpy as np
import glob

Label2Indx = {
            '_unknown_': 0,
            '_silence_': 1,
            '_background_noise_': 1,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11}

class SpeechCommandDataset:
    def __init__(self,data_folder: str, valid_file: str, test_file: str):
        self.data_folder = data_folder
        self.valid_file = valid_file
        self.test_file = test_file
        
    def create_kaldi(self,files, mode='train'):
        os.makedirs('data/'+mode, exist_ok=True)
        with open('data/'+mode+'/wav.scp','w') as f_wav,open('data/'+mode+'/utt2spk','w') as f_u2s, open('data/'+mode+'/spk2utt','w') as f_s2u, open('data/'+mode+'/text','w') as f_txt:
            for filepath in files:
                f_wav.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath+'\n')
                f_u2s.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+'\n')
                f_s2u.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+'\n')
                f_txt.write(filepath.split('/')[-2]+'_'+filepath.split('/')[-1]+' '+filepath.split('/')[-2]+'\n')
            
    def process_data(self,):
        files_list = list()
        train_files = []
        test_files = []
        valid_files = []
        for class_name in Label2Indx.keys():
            if os.path.exists(self.data_folder+'/'+class_name):
                files_list+=glob.glob(self.data_folder+'/'+class_name+'/*.wav')
            else:
                print(f'Folder for class {class_name} does not exist')
        with open(self.valid_file) as fid:
            for line in fid:
                filename = line.rstrip('\n')
                if self.data_folder+'/'+filename in files_list:
                    valid_files.append(self.data_folder+'/'+filename)
                    
        with open(self.test_file) as fid:
            for line in fid:
                filename = line.rstrip('\n')
                if self.data_folder+'/'+filename in files_list:
                    test_files.append(self.data_folder+'/'+filename)
        
        train_files = list(set(files_list) - set(valid_files) -  set(test_files))
        self.create_kaldi(sorted(train_files), mode='train')
        self.create_kaldi(sorted(valid_files), mode='valid')
        self.create_kaldi(sorted(test_files), mode='test')
        
        


if __name__ == '__main__':
    train_folder = '/media/newhd/Google_Speech_Commands/train'
    valid_file = train_folder+'/validation_list.txt'
    test_file = train_folder+'/testing_list.txt'
    dataset = SpeechCommandDataset(train_folder, valid_file, test_file)
    dataset.process_data()