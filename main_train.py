# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:03:15 2018

main_train.py

@author: George
"""

import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from darknet import Darknet
import utils
import cfg
import dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v3 Object Detector with PyTorch')
    parser.add_argument('data', type=str, help='Path to the .data file')
    parser.add_argument('cfg', type=str, help='Path to the .cfg file')
    parser.add_argument('weights', type=str, help='Path to the .weights file')
    parser.add_argument('--save_freq', type=int, default=2)
    args = parser.parse_args()
    return args

def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i

def test(epoch, model, test_loader, use_cuda, conf_thresh, nms_thresh, iou_thresh, eps):
    model.eval()

    num_classes = model.models[-1].num_classes
    anchors     = model.models[-1].anchors
    num_anchors = model.models[-1].num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):

        data = torch.tensor(data)
        if use_cuda:
            data = data.cuda()
        output = model(data).data
        all_boxes = utils.get_region_boxes(output, conf_thresh, num_classes,
                                           anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = utils.nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    utils.logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))



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

def train(epoch, model, region_loss, train_loader, optimizer, use_cuda,
          processed_batches, learning_rate, steps, scales, batch_size,
          save_interval, backupdir ):
    t0 = time.time()
    lr = adjust_learning_rate(optimizer,
                              processed_batches,
                              learning_rate,
                              steps,
                              scales,
                              batch_size)

    utils.logging('epoch %d, processed %d samples, lr %f' % (epoch,
                                                             epoch * len(train_loader.dataset),
                                                             lr))
    model.train()

    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(optimizer,
                             processed_batches,
                             learning_rate,
                             steps,
                             scales,
                             batch_size)

        processed_batches = processed_batches + 1

        t3 = time.time()
        data, target = torch.tensor(data, requires_grad=True), torch.tensor(target, dtype=torch.float32)
        if use_cuda:
            data = data.cuda()
            target = target.cuda()

        t4 = time.time()
        optimizer.zero_grad()

        t5 = time.time()
        #output = model(data, target)
        loss = model(data, target)

        t6 = time.time()
        region_loss.seen = region_loss.seen + data.size(0)
        #loss = region_loss(output, target)

        t7 = time.time()
        loss.backward()

        t8 = time.time()
        optimizer.step()

        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    print('')
    t1 = time.time()
    utils.logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    if (epoch+1) % save_interval == 0:
        utils.logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        model.seen = (epoch + 1) * len(train_loader.dataset)
        model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))

    return processed_batches

def main():

    args = parse_args()
    # Training settings
    datacfg       = args.data
    cfgfile       = args.cfg
    weightfile    = args.weights

    data_options  = utils.read_data_cfg(datacfg)
    net_options   = cfg.parse_cfg(cfgfile)[0]

    trainlist     = data_options['train']
    testlist      = data_options['valid']
    backupdir     = data_options['backup']
    nsamples      = utils.file_lines_win(trainlist)
    gpus          = data_options['gpus']  # e.g. 0,1,2,3
    ngpus         = len(gpus.split(','))
    num_workers   = int(data_options['num_workers'])

    batch_size    = int(net_options['batch']) # darknet batch subdivisions are not supported.
    max_batches   = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]

    #Train parameters
    max_epochs    = max_batches*batch_size/nsamples+1
    use_cuda      = True
    seed          = int(time.time())
    eps           = 1e-5
    save_interval = args.save_freq  # epoches
    dot_interval  = 70  # batches

    # Test parameters
    conf_thresh   = 0.25
    nms_thresh    = 0.4
    iou_thresh    = 0.5

    # Cuda parameters
    cudnn.benchmark = True

    if not os.path.exists(backupdir):
        os.mkdir(backupdir)

    ###############
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    model       = Darknet(cfgfile)

#    if ngpus > 1:
#        model = model.module
#    else:
#        model = model

    region_loss = model.loss

    model.load_weights(weightfile)
    model.print_network()

    region_loss.seen  = model.seen
    processed_batches = model.seen/batch_size

    init_width        = model.width
    init_height       = model.height
    init_epoch        = model.seen/nsamples

    # Initiate data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                       train=True,
                       seen=model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate/batch_size,
                                momentum=momentum,
                                dampening=0,
                                weight_decay=decay*batch_size)

    evaluate = False
    if evaluate:
        utils.logging('evaluating ...')
        test(0, model, test_loader, use_cuda, conf_thresh, nms_thresh, iou_thresh, eps)
    else:
        for epoch in range(init_epoch, int(max_epochs)):
            processed_batches = train(epoch, model, region_loss, train_loader,
                                      optimizer, use_cuda, processed_batches,
                                      learning_rate, steps, scales, batch_size,
                                      save_interval, backupdir)
            test(epoch, model, test_loader, use_cuda, conf_thresh, nms_thresh, iou_thresh, eps)


if __name__ == '__main__':
    main()