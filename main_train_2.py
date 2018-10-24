# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:47 2018

@author: GEO
"""

import os
import argparse
import time
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from Yolov3 import Yolov3Detector, Yolov3ObjectnessClassBBoxCriterion

from utils import non_max_suppression, bbox_ious, load_class_names
import dataset

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
    parser.add_argument('--steps', nargs='+', type=int, default=[40000, 45000])
    parser.add_argument('--max_batches', type=int, default=50200)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.1])
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=2)
    parser.add_argument('--logging', default=False, action='store_true')
    args = parser.parse_args()
    return args

def print_args(args, use_cuda):
    print_and_save('##############', log_file)
    print_and_save('### File things ###', log_file)
    print_and_save('Network file: {}'.format(args.cfg), log_file)
    print_and_save('Pretrained weights: {}'.format(args.weights), log_file)
    print_and_save('Train dataset: {}'.format(args.trainlist), log_file)
    print_and_save('Test dataset: {}'.format(args.testlist), log_file)
    print_and_save('Class names: {}'.format(args.classnames), log_file)
    print_and_save('### Training things ###', log_file)
    print_and_save('Batch size: {}'.format(args.batch_size), log_file)
    print_and_save('Learning rate: {}'.format(args.lr), log_file)
    print_and_save('Momentum: {}'.format(args.momentum), log_file)
    print_and_save('Decay: {}'.format(args.decay), log_file)
    print_and_save('Learning rate steps: {}'.format(args.steps), log_file)
    print_and_save('Learning rate scales: {}'.format(args.scales), log_file)
    print_and_save('Max batches: {}'.format(args.max_batches) if args.max_epochs is None else "Max epochs {}".format(args.max_epochs), log_file)
    print_and_save('### Saving things ###', log_file)
    print_and_save('Output dir: {}'.format(args.output_dir), log_file)
    print_and_save('Eval and save frequency: {}'.format(args.eval_freq), log_file)
    print_and_save('### Hardware things ###', log_file)
    print_and_save('Gpus: {}'.format(args.gpus), log_file)
    print_and_save('Data load workers: {}'.format(args.num_workers), log_file)
    print_and_save('Will use cuda: {}'.format(use_cuda), log_file)
    print_and_save('##############', log_file)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i

def test(epoch, model, modelInfo, test_loader, use_cuda, prev_f=0.0):
    # Test parameters
    conf_thresh   = 0.25
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    model.eval()

    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    
    print_and_save('*****', log_file)
    print_and_save("Evaluating @ epoch {}".format(epoch), log_file)
    for batch_idx, (data, target) in enumerate(test_loader):

        data = torch.tensor(data)
        if use_cuda:
            data = data.cuda()
        print(batch_idx)
        output = model(data)
        output = torch.cat(output, 1)
        boxes = non_max_suppression(output, modelInfo['num_classes'], conf_thresh, nms_thresh)

        for b in range(output.size(0)):
            truths = target[b].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            # this can be done faster. just count the boxes without conf_thresh check
            for i in range(len(boxes[b])):
                if boxes[b][i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = torch.tensor([truths[i][1]-truths[i][3]/2, truths[i][2]-truths[i][4]/2, truths[i][1]+truths[i][3]/2, truths[i][2]+truths[i][4]/2]).cuda()
                best_iou = 0
                best_j = -1
                for j in range(len(boxes[b])):
                    iou = bbox_ious(box_gt, boxes[b][j][:4], x1y1x2y2=True)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[b][best_j][6] == truths[i][0].cuda():
                    correct = correct+1
#        if batch_idx == 1: break

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    print_and_save("precision: {:.3f}, recall: {:.3f}, fscore: {:.3f}".format(precision, recall, fscore), log_file)

    return fscore > prev_f

def adjust_learning_rate(optimizer, batch, learning_rate, steps, scales, batch_size):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch, model, modelInfo, criterion, bce_loss, l1_loss, ce_loss,
          optimizer, train_loader, use_cuda, processed_batches, args):
    lr = adjust_learning_rate(optimizer,
                              processed_batches,
                              args.lr,
                              args.steps,
                              args.scales,
                              args.batch_size)

    print_and_save('*****', log_file)
    print_and_save('epoch {}, processed {} samples, lr {:.4f}'.format(epoch,
                   epoch * len(train_loader.dataset), lr), log_file)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        t0 = time.time()
        adjust_learning_rate(optimizer,
                             processed_batches,
                             lr,
                             args.steps,
                             args.scales,
                             args.batch_size)

        processed_batches = processed_batches + 1

        data, target = torch.tensor(data, requires_grad=True), torch.tensor(target, dtype=torch.float32)
        if use_cuda:
            data = data.cuda()
            target = target.cuda()

#        tf = time.time()
        out = model(data)
#        print('forward',time.time()-tf)
        loss = 0
        for i, output in enumerate(out):
#            tc = time.time()
            loss += criterion(output, target,
                              modelInfo['anchors'][i], modelInfo['masks'][i],
                              modelInfo['num_classes'],
                              modelInfo['lsizes'][i], modelInfo['lsizes'][i],
                              modelInfo['input_shape'][0],
                              modelInfo['input_shape'][1],
                              modelInfo['ignore_thresh'][i],
                              bce_loss, l1_loss, ce_loss)
#            print("crit ",i,time.time()-tc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        print_and_save("Seen {}, loss {}, batch time {:.3f} s".format(processed_batches*args.batch_size, loss, t1-t0), log_file)
        #print_and_save('', log_file)
#        if batch_idx == 5: break
    return processed_batches

def main():
    global log_file
    args = parse_args()
    cfgfile       = args.cfg
    weightfile    = args.weights
    classnames    = load_class_names(args.classnames)

    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    log_file = os.path.join(args.output_dir,"{}_log.txt".format('train')) if args.logging==True else None
    print_args(args, use_cuda)

    ###############
    seed = int(time.time())
    torch.manual_seed(time.time())
    if use_cuda:
        torch.cuda.manual_seed(seed)

    model = Yolov3Detector(cfgfile, (416, 416))

    print_and_save(model, log_file)
    print_and_save("Trainable parameters {}".format(count_parameters(model)), log_file)
    model.load_weights(weightfile)
    print_and_save('Loading weights from {}... Done!'.format(weightfile), log_file)
    yolo_inds = model.yolo_inds
    yolo_sizes = [13,26,52]
    yolo_anchors = [model.detector[i].yolo.anchors for i in yolo_inds]
    yolo_masks = [model.detector[i].yolo.mask for i in yolo_inds]
    ignore_thresh = [model.detector[i].yolo.ignore_thresh for i in yolo_inds]
    modelInfo = {'inds':yolo_inds, 'lsizes':yolo_sizes, 'input_shape':(416,416),
                 'anchors':yolo_anchors, 'masks':yolo_masks,
                 'ignore_thresh': ignore_thresh,
                 'num_classes': len(classnames)}

    l1_loss = nn.L1Loss(size_average=False, reduce=False)
    bce_loss = nn.BCELoss(size_average=False, reduce=False)
    ce_loss = nn.CrossEntropyLoss(size_average=False, reduce=False)
    objclasscriterion = Yolov3ObjectnessClassBBoxCriterion

     # Initiate data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(args.trainlist, shape=(416, 416),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                       train=True,
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    max_epochs = args.max_batches * args.batch_size/len(train_loader)+1 if args.max_epochs is None else args.max_epochs
#    print_and_save("Training for {} epochs".format(max_epochs), log_file)
    
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(args.testlist, shape=(416, 416),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), train=False),
        batch_size=len(args.gpus),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    processed_batches = model.seen/args.batch_size

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': args.decay*args.batch_size}]
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr/args.batch_size,
                                momentum=args.momentum,
                                dampening=0,
                                weight_decay=args.decay*args.batch_size)

    if use_cuda:
        device = 'cuda:{}'.format(args.gpus[0])
        l1_loss = l1_loss.cuda()
        bce_loss = bce_loss.cuda()
        ce_loss = ce_loss.cuda()
        if len(args.gpus) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(device)
            
    evaluate = False
    if evaluate:
        print('evaluating ...')
        test(0, model, modelInfo, test_loader, use_cuda)
    else:
        prev_f = 0.0
        for epoch in range(int(max_epochs)):
            processed_batches = train(epoch, model, modelInfo,
                                      objclasscriterion,
                                      bce_loss, l1_loss, ce_loss,
                                      optimizer,
                                      train_loader, use_cuda,
                                      processed_batches, args)
            f_score = test(epoch, model, modelInfo, test_loader, use_cuda, prev_f)
            if (epoch+1) % args.eval_freq == 0:
                print_and_save('Saving weights to {}/{:06d}.weights'.format(args.output_dir, epoch+1), log_file)
                model.seen = (epoch + 1) * len(train_loader.dataset)
                name = os.path.join(args.output_dir,'epoch_{:06d}'.format(epoch+1))
                try:
                    model.save_weights(name+'.weights')
                except:
                    print('saving data parallel')
                    model.module.save_weights(name+'.weights')
                if f_score > prev_f:
                    shutil.copyfile(name+'.weights',
                                    os.path.join(args.output_dir, 'best.weights'))
            prev_f = f_score

if __name__ == '__main__':
    main()