# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:48:47 2018

@author: GEO
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from Yolov3 import Yolov3Detector, Yolov3ObjectnessClassBBoxCriterion

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
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--steps', nargs='+', type=int, default=[40000, 45000])
    parser.add_argument('--max_batches', type=int, default=50200)
    parser.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.1])
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
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
    print('Learning rate steps: {}'.format(args.steps))
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

def test(epoch, model, test_loader, use_cuda, conf_thresh, nms_thresh, iou_thresh, eps):
    model.eval()

    num_classes = model.detector[-1].num_classes
    anchors     = model.detector[-1].anchors
    num_anchors = model.models[-1].num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):

        data = torch.tensor(data)
        if use_cuda:
            data = data.cuda()
        output = model(data)

        for i in range(output.size(0)):
            #boxes = all_boxes[i]
            boxes = output[i]
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
    print("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

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

def train(epoch, model, criterion, bce_loss, l1_loss, optimizer, train_loader,
          use_cuda, processed_batches, args):
    lr = adjust_learning_rate(optimizer,
                              processed_batches,
                              args.lr,
                              args.steps,
                              args.scales,
                              args.batch_size)

    print('epoch %d, processed %d samples, lr %f' % (epoch,
                                                     epoch * len(train_loader.dataset),
                                                     lr))
    model.train()
    yolo_inds = model.yolo_inds
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

        out = model(data)
        loss = 0
        for i, output in enumerate(out):
            loss += criterion(output, target,
                             model.detector[yolo_inds[i]].yolo.anchors,
                             model.detector[yolo_inds[i]].yolo.mask,
                             20,
                             model.detector[yolo_inds[i]].yolo.layer_height,
                             model.detector[yolo_inds[i]].yolo.layer_width,
                             416, 416,
                             model.detector[yolo_inds[i]].yolo.ignore_thresh,
                             bce_loss, l1_loss)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()
        print("Epoch {}, loss {}, time {:.3f}".format(epoch, loss, t1-t0))

    if (epoch+1) % args.eval_freq == 0:
        print('save weights to %s/%06d.weights' % (args.output_dir, epoch+1))
        model.seen = (epoch + 1) * len(train_loader.dataset)
        model.save_weights('%s/%06d.weights' % (args.output_dir, epoch+1))

    return processed_batches

def main():

    args = parse_args()
    cfgfile       = args.cfg
    weightfile    = args.weights

    use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    log_file = "{}_log.txt".format('train')
    print_args(args, use_cuda)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ###############
    seed = int(time.time())
    torch.manual_seed(time.time())
    if use_cuda:
        torch.cuda.manual_seed(seed)

    model = Yolov3Detector(cfgfile, (416, 416))
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(count_parameters(model))
    model.load_weights(weightfile)
#    yolo_inds = model.yolo_inds


#    x = torch.rand((1,3,416,416))
#    out = model(x)
#    targets = torch.tensor([[14, 0.509915014164306, 0.51, 0.9745042492917847, 0.972],[11., 0.34419263456090654, 0.611, 0.4164305949008499, 0.262]]).unsqueeze(0)

    l1_loss = nn.L1Loss(size_average=True, reduce=True)
#    bboxcriterion = Yolov3BboxCriterion
    bce_loss = nn.BCEWithLogitsLoss(size_average=True, reduce=True)
#    classcriterion = Yolov3ClassCriterion
#    objectnesscriterion = Yolov3ObjectnessCriterion
    objclasscriterion = Yolov3ObjectnessClassBBoxCriterion

#    loss_bbox = bboxcriterion(out[0], targets,
#                              model.detector[yolo_inds[0]].yolo_1.anchors,
#                              model.detector[yolo_inds[0]].yolo_1.mask,
#                              13, 13, 416, 416, l1_loss)

#    loss_class = classcriterion(out[0], targets,
#                                model.detector[yolo_inds[0]].yolo_1.anchors,
#                                model.detector[yolo_inds[0]].yolo_1.mask,
#                                20, 13, 13, 416, 416, bce_loss)

#    loss_obj = objectnesscriterion(out[0], targets,
#                                   model.detector[yolo_inds[0]].yolo_1.mask,
#                                   3, 3)

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
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    max_epochs = args.max_batches * args.batch_size/len(train_loader)+1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(args.testlist, shape=(416, 416),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    processed_batches = model.seen/args.batch_size
    if use_cuda:
        if len(args.gpus) > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

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

    evaluate = False
    if evaluate:
        print('evaluating ...')
        #test(0, model, test_loader, use_cuda, conf_thresh, nms_thresh, iou_thresh, eps)
    else:
        for epoch in range(int(max_epochs)):
            processed_batches = train(epoch, model, objclasscriterion,
                                      bce_loss, l1_loss, optimizer,
                                      train_loader, use_cuda,
                                      processed_batches, args)
            #test(epoch, model, test_loader, use_cuda, conf_thresh, nms_thresh, iou_thresh, eps)


if __name__ == '__main__':
    main()