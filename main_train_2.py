# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:47 2018

@author: GEO
"""

import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn

from utils import read_data_cfg
from cfg import parse_cfg

def print_and_save(text, path):
    print(text)
    if path is not None:
        with open(path, 'a') as f:
            print(text, file=f)
            
def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v3 Object Detector with PyTorch')
    parser.add_argument('cfg', type=str, help='Path to the .cfg file')
    parser.add_argument('weights', type=str, help='Path to the .weights file')
    parser.add_argument('trainlist', type=str)
    parser.add_argument('testlist', type=str)
    parser.add_argument('classnames', type=str)
    parser.add_argument('--output_dir', type=str, default=r'backup/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--lr_steps', nargs='+', type=int, default=[40000, 45000])
    parser.add_argument('--max_batches', type=int, default=50200)
    parser.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.1])
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=2)
    args = parser.parse_args()
    return args

def print_args(args, use_cuda):
    print('##############')
    print('### File things ###')              
    print('Network file: {}'.format(args.cfg))
    print('Pretrained weights: {}'.format(args.weights))
    print('Train dataset: {}'.format(args.trainlist))
    print('Test dataset: {}'.format(args.testlist))
    print('Class names: {}'.format(args.classnames))
    print('### Training things ###')
    print('Batch size: {}'.format(args.batch_size))
    print('Learning rate: {}'.format(args.lr))
    print('Momentum: {}'.format(args.momentum))
    print('Decay: {}'.format(args.decay))
    print('Learning rate steps: {}'.format(args.lr_steps))
    print('Learning rate scales: {}'.format(args.scales))
    print('Max batches: {}'.format(args.max_batches))
    print('### Saving things ###')
    print('Output dir: {}'.format(args.output_dir))
    print('Eval and save frequency: {}'.format(args.eval_freq))
    print('### Hardware things ###')
    print('Gpus: {}'.format(args.gpus))
    print('Data load workers: {}'.format(args.num_workers))
    print('Will use cuda: {}'.format(use_cuda))
    print('##############')
    
def main():
    
    args = parse_args()
    cfgfile       = args.cfg
    weightfile    = args.weights
    
    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    
    log_file = "{}_log.txt".format('train')   
    print_args(args, use_cuda)
    
    
    
    ###############
    torch.manual_seed(time.time())
    if use_cuda:
        torch.cuda.manual_seed(seed)
        
    # TODO: calculate max epochs by dividing trainset length / max batches arg
    
    print('end')
    

    
    
    

if __name__ == '__main__':
    main()