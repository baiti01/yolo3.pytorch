# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:14:58 2018

@author: Γιώργος
"""

import numpy as np
#from collections import OrderedDict
import torch
import torch.nn as nn

from utils import bbox_ious
from cfg import load_conv, load_conv_bn, save_conv, save_conv_bn

def parse_cfgfile(cfgfile):
    '''
    source: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/parse_config.py
    '''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x.rstrip().lstrip() for x in lines if x and not x.startswith('#')] # get rid of comments and empty lines

    module_defs = []
    for line in lines:
        if line.startswith('['): # this marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.strip().split('=')
            module_defs[-1][key.strip()] = value

    return module_defs

def create_modules(modules_list, net_input_size):
    hyperparams = modules_list.pop(0) # get [net]
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    conv_id = 0
    upsample_id = 0
    route_id = 0
    shortcut_id = 0
    #yolo_id = 0
    yolo_inds = []
    for i, module in enumerate(modules_list):
        modules = nn.Sequential()
        if module['type'] == 'convolutional':
            conv_id += 1
            bn = int(module['batch_normalize']) if 'batch_normalize' in module.keys() else 0
            filters = int(module['filters'])
            kernel_size = int(module['size'])
            pad = (kernel_size - 1)//2 if int(module['pad']) else 0
            stride = int(module['stride'])
            modules.add_module(
                    "conv",
                    nn.Conv2d(
                            in_channels=output_filters[-1],
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=pad,
                            bias=not bn,
                            ))
            if bn:
                modules.add_module("batch_norm",
                                   nn.BatchNorm2d(filters))
            if module['activation'] == 'leaky':
                modules.add_module("leaky",
                                   nn.LeakyReLU(0.1, inplace=True))
            elif module['activation'] == 'relu':
                modules.add_module("relu",
                                   nn.ReLU(inplace=True))
        elif module['type'] == 'maxpool':
            pass
        elif module['type'] == 'upsample':
            upsample_id += 1
            modules.add_module("upsample",
                               nn.Upsample(scale_factor=int(module['stride'])))
        elif module['type'] == 'route':
            route_id += 1
            layers = [int(x) for x in module['layers'].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route", EmptyLayer())
        elif module['type'] == 'shortcut':
            shortcut_id += 1
            filters = output_filters[int(module['from'])]
            modules.add_module("shortcut", EmptyLayer())
        elif module['type'] == 'yolo':
            #yolo_id += 1
            yolo_inds.append(len(module_list))
            mask = [int(x) for x in module['mask'].split(',')]
            anchors = np.array([float(x) for x in module['anchors'].split(',')]).reshape(-1, 2)
            classes = int(module['classes'])
            ignore_thresh = float(module['ignore_thresh'])
            truth_thresh = float(module['truth_thresh'])
            # these two should be put to network hyperparameters
            jitter = float(module['jitter'])
            random_size = int(module['random'])
            modules.add_module("yolo",
                               Yolov3Layer(anchors, mask, classes, net_input_size,
                                           jitter, ignore_thresh, truth_thresh,
                                           random_size))
#            modules = nn.Sequential(OrderedDict([
#                        ("yolo_{}".format(yolo_id),
#                         Yolov3Layer(anchors, mask, classes, net_input_size,
#                                     jitter, ignore_thresh, truth_thresh,
#                                     random_size))
#                        ]))
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list, yolo_inds

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x
#import time
class Yolov3Layer(nn.Module):
    def __init__(self, anchors, mask, classes, net_input_size, jitter, ignore_thresh,
                 truth_thresh, random_size):
        super(Yolov3Layer, self).__init__()
        self.anchors = anchors
        self.mask = mask
        self.classes = classes
        self.net_input_size = net_input_size
        self.ignore_thresh = ignore_thresh
        self.bbox_attributes = 5 + classes

        self.layer_anchors = self.anchors[mask, :]
    # yolo forward should just be about reshaping and making the boxes
    # in order to move the loss criterion out of the forward function
    def forward(self, output):

        batch_size = output.size(0)
        self.layer_height = output.size(2)
        self.layer_width = output.size(3)
        num_anchors = len(self.mask)

        prediction = output.view(batch_size, num_anchors, self.bbox_attributes,
                                 self.layer_height, self.layer_width
                                 ).permute(0, 1, 3, 4, 2).contiguous()
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        width = prediction[..., 2]  # Width
        height = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:]) # Cls pred.
#        pred_conf = prediction[..., 4]  # Conf
#        pred_cls = prediction[..., 5:] # Cls pred.
#        t1 = time.time()
        yolo_boxes = self.get_yolo_boxes_fast(x,y,width,height, self.layer_anchors)
#        t2 = time.time()
#        yolo_boxes = self.get_yolo_boxes(x, y, width, height, self.layer_anchors).cuda()
#        t3 = time.time()
#        print('{:.3f} to make {} yolo boxes'.format(t2-t1, yolo_boxes.numel()//4))
#        print('{:.3f} to make {} fast yolo boxes'.format(t3-t2, yolo_boxes_2.numel()//4))
        output = torch.cat(
                (
                        yolo_boxes.view(batch_size, -1, 4),
                        pred_conf.view(batch_size, -1, 1),
                        pred_cls.view(batch_size, -1, self.classes),
                ),
                -1,
                )

        return output

    def get_yolo_boxes_fast(self, x, y, width, height, masked_anchors):
        assert x.shape == y.shape == width.shape == height.shape
        net_width = self.net_input_size[1]
        net_height = self.net_input_size[0]
        batches, num_anchors, grid_w, grid_h = x.shape

        grid_ij = torch.linspace(0, grid_w-1, grid_w).repeat(grid_h,1).view(1,1,grid_h, grid_w).cuda()

        masked_anchors_w = torch.tensor(masked_anchors[:,0]).type(torch.float).cuda()
        masked_anchors_w = masked_anchors_w.repeat(grid_w, 1)
        masked_anchors_w = masked_anchors_w.repeat(grid_h, 1, 1)
        masked_anchors_w = masked_anchors_w.repeat(batches, 1, 1, 1)
        masked_anchors_w = masked_anchors_w.permute(0,3,1,2)
        masked_anchors_h = torch.tensor(masked_anchors[:,1]).type(torch.float).cuda()
        masked_anchors_h = masked_anchors_h.repeat(grid_w, 1)
        masked_anchors_h = masked_anchors_h.repeat(grid_h, 1, 1)
        masked_anchors_h = masked_anchors_h.repeat(batches, 1, 1, 1)
        masked_anchors_h = masked_anchors_h.permute(0,3,1,2)
        new_x = (grid_ij + x)/grid_w
        new_y = (grid_ij.transpose(2,3) + y)/grid_h
        new_width = torch.exp(width) * masked_anchors_w / net_width
        new_height = torch.exp(height) * masked_anchors_h / net_height

        return torch.cat([new_x.unsqueeze(-1),
                          new_y.unsqueeze(-1),
                          new_width.unsqueeze(-1),
                          new_height.unsqueeze(-1)],
                          -1)

    def get_yolo_boxes(self, x, y, width, height, masked_anchors):
        assert x.shape == y.shape == width.shape == height.shape
        net_width = self.net_input_size[1]
        net_height = self.net_input_size[0]

        batches, num_anchors, grid_w, grid_h = x.shape
        boxes = []
        for b in range(batches):
            for a in range(num_anchors):
                for j in range(grid_h):
                    for i in range(grid_w):
                        new_x = (i + x[b][a][j][i])/grid_w
                        new_y = (j + y[b][a][j][i])/grid_h
                        new_width = torch.exp(width[b][a][j][i]) * masked_anchors[a][0] / net_width
                        new_height = torch.exp(height[b][a][j][i]) * masked_anchors[a][1] / net_height
                        boxes.append([new_x, new_y, new_width, new_height])
        return torch.tensor(boxes).view(batches, num_anchors, grid_w, grid_h, 4)#.cuda()

def Yolov3ObjectnessClassBBoxCriterion(inputs, targets, anchors, anchor_mask, num_classes,
                                   layer_height, layer_width, net_height, net_width,
                                   ignore_thresh, bce_loss, l1_loss, ce_loss):
    batch_size = inputs.size(0)
    inputs = inputs.view(batch_size, len(anchor_mask), layer_height, layer_width, -1)
    targets = targets.view(batch_size, -1, 5)

    best_iou_mask = torch.zeros(batch_size, len(anchor_mask), layer_height, layer_width).cuda()
    ignore_mask = torch.zeros_like(best_iou_mask)
    tx = torch.zeros_like(best_iou_mask).cuda()
    ty = torch.zeros_like(best_iou_mask).cuda()
    tw = torch.zeros_like(best_iou_mask).cuda()
    th = torch.zeros_like(best_iou_mask).cuda()
    ts = torch.zeros_like(best_iou_mask).cuda()
    num_truths = 0
    loss_class = torch.zeros(1).cuda()
    for b in range(batch_size):
        preds = inputs[b][...,:4]
        for t in targets[b]:
            if t.sum() == 0:
                continue
            num_truths += 1 # keep count of ground truth boxes over the batch

            # get the ious of the predictions with the current ground truth box
            pred_ious = bbox_ious(preds.permute(3,0,1,2), t[1:], False)
            # ignore objectness loss from boxes with decent iou with any of the
            # gt objects of the currently evaluated image
            ignore_mask[b] = torch.max(ignore_mask[b],
                       torch.tensor(pred_ious > ignore_thresh, dtype=torch.float).cuda())

            # get gt box's corresponding grid cell
            gt_i = int(t[1] * layer_width)
            gt_j = int(t[2] * layer_height)

            # get gt box's corresponding anchor
            # - could be non existing if the gt could be predicted better
            # at another yolo_layer
            truth = torch.tensor([0, 0, t[3], t[4]])#.cuda()
            best_anchor = -1
            best_iou = 0
            for i, anc in enumerate(anchors):
                anchor_box = torch.tensor([0,0,
                                           anc[0]/net_width,
                                           anc[1]/net_height])#.cuda()
                iou = bbox_ious(truth, anchor_box, False)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            # if the gt is for the anchors of this layer calculate bbox and class loss
            if best_anchor in anchor_mask:
                best_anchor_norm = best_anchor % len(anchor_mask)
                # keep best box prediction for class and bbox loss
                best_iou_mask[b][best_anchor_norm][gt_j][gt_i] = 1

                tx[b][best_anchor_norm][gt_j][gt_i] = t[1] * layer_width - gt_i
                ty[b][best_anchor_norm][gt_j][gt_i] = t[2] * layer_height - gt_j
                tw[b][best_anchor_norm][gt_j][gt_i] = torch.log(t[3] * net_width / anchors[best_anchor][0])
                th[b][best_anchor_norm][gt_j][gt_i] = torch.log(t[4] * net_height / anchors[best_anchor][1])
                ts[b][best_anchor_norm][gt_j][gt_i] = 2 * t[2] * t[3]

                target_pred = inputs[b][best_anchor_norm][gt_j][gt_i][5:]
                one_hot = torch.zeros(num_classes).cuda()
                one_hot[int(t[0])] = 1.
                for c in range(num_classes):
                    temp_loss = bce_loss(target_pred[c], one_hot[c])
                    loss_class += temp_loss.cuda()
                #loss_class += ce_loss(target_pred.unsqueeze(0), t[0].unsqueeze(0).type(torch.long))

    loss_x = torch.zeros(1).cuda()
    loss_y = torch.zeros(1).cuda()
    loss_width = torch.zeros(1).cuda()
    loss_height = torch.zeros(1).cuda()
    #loss_class = torch.zeros(1)
    loss_objectness = torch.zeros(1).cuda()

    loss_x = ts * l1_loss(inputs[...,0] * best_iou_mask, tx)
    loss_y = ts * l1_loss(inputs[...,1] * best_iou_mask, ty)
    loss_width = ts * l1_loss(inputs[...,2] * best_iou_mask, tw)
    loss_height = ts * l1_loss(inputs[...,3] * best_iou_mask, th)
    loss_class = torch.sum(loss_class) / num_truths
#    for c in range(num_classes):
#        temp_loss = bce_loss(inputs[...,5+c] * best_iou_mask, targets[...,0]==c)
#        loss_class += temp_loss
    ignore_mask = torch.max(ignore_mask, best_iou_mask) # to ignore the box with positive loss
    loss_objectness_wrong = l1_loss(inputs[...,4]*(1-ignore_mask), torch.zeros_like(ignore_mask))
    loss_objectness_right =  l1_loss(inputs[...,4]*best_iou_mask, torch.ones_like(ignore_mask)*best_iou_mask)
    nz_x = np.count_nonzero(loss_x.detach().cpu().numpy())
    if nz_x == 0: nz_x += 1
    nz_y = np.count_nonzero(loss_y.detach().cpu().numpy())
    if nz_y == 0: nz_y += 1
    nz_width = np.count_nonzero(loss_width.detach().cpu().numpy())
    if nz_width == 0: nz_width += 1
    nz_height = np.count_nonzero(loss_height.detach().cpu().numpy())
    if nz_height == 0: nz_height += 1
    nz_obj_wr = np.count_nonzero(loss_objectness_wrong.detach().cpu().numpy())
    if nz_obj_wr == 0: nz_obj_wr += 1
    nz_obj_ri = np.count_nonzero(loss_objectness_right.detach().cpu().numpy())
    if nz_obj_ri == 0: nz_obj_ri += 1
    loss_x = torch.sum(torch.pow(loss_x,2))/nz_x
    loss_y = torch.sum(torch.pow(loss_y,2))/nz_y
    loss_width = torch.sum(torch.pow(loss_width,2))/nz_width
    loss_height = torch.sum(torch.pow(loss_height,2))/nz_height
    loss_objectness_wrong = torch.sum(torch.pow(loss_objectness_wrong,2))/nz_obj_wr
    loss_objectness_right = torch.sum(torch.pow(loss_objectness_right,2))/nz_obj_ri
    print("losses: x {:.3f}, y {:.3f}, w {:.3f}, h {:.3f}, class {:.3f}, obj no {:.3f}, obj yes {:.3f}".format(loss_x, loss_y, loss_width, loss_height, loss_class, loss_objectness_wrong, loss_objectness_right))
    return loss_x + loss_y + loss_width + loss_height + loss_class + loss_objectness_wrong + loss_objectness_right


def Yolov3ObjectnessCriterion(inputs, targets, anchor_mask, layer_height, layer_width):
    batch_size = inputs.size(0)
    inputs = inputs.view(batch_size, len(anchor_mask), layer_height, layer_width, -1)
    targets = targets.view(batch_size, -1, 5)

    loss_objectness = torch.zeros(batch_size, len(anchor_mask), layer_height, layer_width)#.cuda()
    for b in range(batch_size):
        preds = inputs[b][...,:4]
        confs = inputs[b][...,4]
        best_ious = torch.zeros(batch_size, len(anchor_mask), layer_height, layer_width)#.cuda()
        for t in targets[b]:
            if t.sum() == 0:
                continue
            ious = bbox_ious(preds.permute(3,0,1,2), t[1:], False)
            best_ious = torch.max(best_ious, ious)
        loss_objectness[b] = -confs[b] # we want loss to be 1 when a box is not with 100% confidence and 0

    return loss_objectness

def Yolov3ClassCriterion(inputs, targets, anchors, mask, num_classes,
                         layer_height, layer_width,
                         net_height, net_width, bce_loss):
    batch_size = inputs.size(0)
    targets = targets.view(batch_size, -1, 5)
    inputs = inputs.view(batch_size, len(mask), layer_height, layer_width, -1)

    loss_class = torch.zeros(1)

    #bce2 = nn.BCELoss(size_average=False, reduce=False)
#    softmax = nn.Softmax(dim=0)
#    ce = nn.CrossEntropyLoss()
    num_truths = 0
    for b in range(batch_size):
        for t in targets[b]:
            if t.sum() == 0:
                continue
            num_truths += 1

            gt_i = int(t[1] * layer_width)
            gt_j = int(t[2] * layer_height)

            truth = torch.tensor([0, 0, t[3], t[4]])#.cuda()
            best_anchor = -1
            best_iou = 0
            for i, anc in enumerate(anchors):
                anchor_box = torch.tensor([0,0,
                                           anc[0]/net_width,
                                           anc[1]/net_height])#.cuda()
                iou = bbox_ious(truth, anchor_box, False)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            if best_anchor in mask:
                best_anchor_norm = best_anchor % len(mask)
                target_pred = inputs[b][best_anchor_norm][gt_j][gt_i][5:]

                one_hot = torch.zeros(num_classes)#.cuda()
                one_hot[int(t[0])] = 1.
                for c in range(num_classes):
                    temp_loss = bce_loss(target_pred[c], one_hot[c])
                    #temp_loss_2 = bce2(torch.sigmoid(target_pred[c]), one_hot[c])
                    #print(temp_loss, temp_loss_2)
                    loss_class += temp_loss

#                ce_loss = ce(target_pred.unsqueeze(0), torch.tensor(t[0].unsqueeze(0), dtype=torch.long))

#        print(loss_class, ce_loss)

        loss_class /= num_truths

        return loss_class



def Yolov3BboxCriterion(inputs, targets, anchors, mask,
                        layer_height, layer_width,
                        net_height, net_width, l1_loss):
    batch_size = inputs.size(0)

    targets = targets.view(batch_size, -1, 5)

    # inputs is batchsize x num_boxes x 4
    # anchors is 9x2, masked_anchors is list of indices with len 3

    inputs = inputs.view(batch_size, len(mask), layer_height, layer_width, -1)
    loss_x = torch.zeros(1)#.cuda()
    loss_y = torch.zeros(1)#.cuda()
    loss_width = torch.zeros(1)#.cuda()
    loss_height = torch.zeros(1)#.cuda()

    num_truths = 0
    for b in range(batch_size):
        for t in targets[b]:
            if t.sum() == 0:
                continue
            num_truths += 1

            gt_i = int(t[1] * layer_width)
            gt_j = int(t[2] * layer_height)

            truth = torch.tensor([0, 0, t[3], t[4]])#.cuda()
            best_anchor = -1
            best_iou = 0
            for i, anc in enumerate(anchors):
                anchor_box = torch.tensor([0,0,
                                           anc[0]/net_width,
                                           anc[1]/net_height])#.cuda()
                iou = bbox_ious(truth, anchor_box, False)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            if best_anchor in mask:
                best_anchor_norm = best_anchor % len(mask)
                target_pred = inputs[b][best_anchor_norm][gt_j][gt_i][:4]
                iou = bbox_ious(t[1:], target_pred, False)

                tx = t[1] * layer_width - gt_i
                ty = t[2] * layer_height - gt_j
                tw = torch.log(t[3] * net_width / anchors[best_anchor][0])
                th = torch.log(t[4] * net_height / anchors[best_anchor][1])
                scale = 2 * t[2] * t[3]

                loss_x += scale * l1_loss(target_pred[0], tx)
                loss_y += scale * l1_loss(target_pred[1], ty)
                loss_width += scale * l1_loss(target_pred[2], tw)
                loss_height += scale * l1_loss(target_pred[3], th)

        loss_x /= num_truths
        loss_y /= num_truths
        loss_width /= num_truths
        loss_height /= num_truths
        return loss_x + loss_y + loss_width + loss_height

class Yolov3Loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, anchors, target):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

class Yolov3Detector(nn.Module):
    def __init__(self, cfgfile, net_input_size):
        super(Yolov3Detector, self).__init__()
        self.modules_list = parse_cfgfile(cfgfile)
        self.hyperparams, self.detector, self.yolo_inds = create_modules(
                self.modules_list,
                net_input_size)
        self.seen = 0

    def forward(self, x):

        layer_outputs = []
        yolo_outputs = []
        for i, (nnmodule, module_name) in enumerate(zip(self.detector, self.modules_list)):
            if module_name['type'] == 'convolutional' \
                              or module_name['type'] == 'upsample' \
                              or module_name['type'] == 'maxpool':
                x = nnmodule(x)
            elif module_name['type'] == 'route':
                layers = [int(x) for x in module_name['layers'].split(',')]
                x = torch.cat([layer_outputs[l] for l in layers], 1)
            elif module_name['type'] == 'shortcut':
                layer_from = int(module_name['from'])
                x = layer_outputs[-1] + layer_outputs[layer_from]
            elif module_name['type'] == 'yolo':
                x = nnmodule(x)
                yolo_outputs.append(x)
            layer_outputs.append(x)

        return yolo_outputs

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        for i, (nnmodule, module_name) in enumerate(zip(self.detector, self.modules_list)):
            if start >= buf.size:
                break
            if module_name['type'] == 'net':
                continue
            elif module_name['type'] == 'convolutional':
                batch_normalize = int(module_name['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, nnmodule.conv, nnmodule.batch_norm)
                else:
                    start = load_conv(buf, start, nnmodule.conv)

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.detector)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.modules_list[blockId]
            if block['type'] == 'convolutional':
                nnmodule = self.detector[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, nnmodule.conv, nnmodule.batch_norm)
                else:
                    save_conv(fp, nnmodule.conv)
        fp.close()