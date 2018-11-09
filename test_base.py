import argparse
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets

import csv

from Utils.logger import Logger
from Models.inceptionv4 import inception_v4
from Models.inceptionv3 import inception_v3
from Data.data_loader import get_dataset

cudnn.benchmark = True
# from create_dataset import get_dataset

epochs = 1000
batch_size = 128
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4
print_freq = 1
adjust_fr = 200
sgd_switch = True
save_filename = './ckpt/incept_0.pth.tar'
validir = '/data/ABUS_cache/Stomach_cancer/data/validate'
traindir = '/data/ABUS_cache/Stomach_cancer/data/train'

# mean_0:0.89767712104785
# mean_1: 0.7683625751988915
# mean_2:0.8887981215775167
#
# std_0:0.09704974462549215
# std_1: 0.15829053693288206
# std_2:0.11856540180557106
# ============================================================================================
# ============================================================================================
# ============================================================================================
# data loading

print('loading dataaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

normalize = transforms.Normalize(mean=[0.898, 0.768, 0.888],
                                 std=[0.098, 0.158, 0.119])
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(299),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(brightness=0.1, saturation=0.5, contrast=0.5, hue=0.05),
                                      transforms.ToTensor(),
                                      normalize
                                      ])
val_transform = transforms.Compose([transforms.Resize(299),
                                    transforms.ToTensor(),
                                    #  transforms.CenterCrop(224),
                                    normalize
                                    ])

# train_dataset, val_dataset = get_dataset(train_transform, val_transform)
train_dataset = datasets.ImageFolder(traindir, train_transform)
val_dataset = datasets.ImageFolder(validir, val_transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size
                                           , shuffle=True, pin_memory=True, num_workers=12)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True, num_workers=12)
# ============================================================================================
# ============================================================================================
# ============================================================================================
# parse args

parser = argparse.ArgumentParser(description='StomachCancer Training')

parser.add_argument('--resume', default='', type=str, metavar='path',
                    help='path to latest checkpoint')
parser.add_argument('--log', default='log/0', type=str, metavar='logpath',
                    help='the number of each test log')

# 3 stand for inception_v3, type other will train with inception_v4
parser.add_argument('--model', default='3', type=str, metavar='inception_model',
                    help='choose the inception model')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pretrained model')



# ============================================================================================
# ============================================================================================
# ============================================================================================
# checkpoint || Tracking || adjust_learning_rate || accuracy

def save_checkpoint(state):
    torch.save(state, save_filename)


class Tracking(object):

    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, label, mining_save=False, csv_wr=None, inputs=None, i=None):
    b_size = label.size(0)
    with torch.no_grad():
        _, pred = output.max(1)
        check = pred.eq(label)
        prec = check.sum(0)
        accur = prec.item() / b_size
        if mining_save:
            for h in range(len(check)):
                if check[h] == 0:
                    csv_wr.writerow([inputs[h], label[h], i, h])
    return accur


# ============================================================================================
# ============================================================================================
# ============================================================================================
# main

def main():
    global best_prec1, args
    args = parser.parse_args()
    logger = Logger(args.log)

    # using inceptionV3 or inception V4
    if args.model == '3':
        model = inception_v3(num_classes=2).cuda()
    else:
        model = inception_v4(num_classes=2).cuda()

    # loading the different pretrained model
    if args.pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys
        pattern = re.compile(
            r''
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        if args.model == '3':
            state_dict = torch.load('inception_v3.pth')
            print('using the pretrained inception_v3 model')
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            model.load_state_dict(state_dict, False)
        else:
            state_dict = torch.load('inception_v4.pth')
            print('using the pretrained inception_v4 model')
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "classifier" not in k}
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            model.load_state_dict(state_dict, False)

    model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # switch to adam
    if sgd_switch:
        optimizer = torch.optim.SGD(model.parameters(),
                                    learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=0.9,
                                    nesterov=False)


    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     learning_rate,
                                     weight_decay=weight_decay,
                                     amsgrad=True)

    # resume
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            #            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # epoch iteration
    for epoch in range(start_epoch, epochs):
        # for sgd optimizer, turn on the learning rate adjusting
        if sgd_switch:
            adjust_learning_rate(optimizer, epoch)
        print('============ EPOCH ' + str(epoch) + ' ====================')

        # training
        print('==> training')

        train(train_loader, model, criterion, optimizer, epoch, logger)

        # validating
        print('==> validating')
        csv_file = open('inception_val_mining.csv', 'w', encoding='utf-8')
        csv_wr = csv.writer(csv_file)
        validate(val_loader, model, criterion, epoch, logger, csv_wr)
        csv_file.close()
        # saveing
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        })


# ============================================================================================
# ============================================================================================
# ============================================================================================
# train
def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = Tracking()
    losses = Tracking()
    precision = Tracking()

    model.train()
    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs.cuda()
        # targets = torch.LongTensor(np.array(targets)).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output and accuracy
        output = model(inputs)
        if args.model == '3':
            loss = sum(criterion(out, targets) for out in output)
            accu = accuracy(output[0], targets)
        else:
            loss = criterion(output, targets)
            accu = accuracy(output, targets)

        # update accurcy and loss
        losses.update(loss.item(), inputs.size(0))
        precision.update(accu, inputs.size(0))

        # compute gradient and loss backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure batch time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  '|| Batch_Time: {batch_time.avg:.3f}'
                  '|| Loss:  {losses.avg:.3f}'
                  '|| accuracy: {precision.avg:.3f}'
                .format(
                epoch, i, len(train_loader)
                , batch_time=batch_time
                , losses=losses
                , precision=precision))
    logger.scalar_summary('train_loss', losses.avg, epoch)
    logger.scalar_summary('train_prec', precision.avg, epoch)


# ============================================================================================
# ============================================================================================
# ============================================================================================
# validate

def validate(val_loader, model, criterion, epoch, logger, csv_wr):
    batch_time = Tracking()
    losses = Tracking()
    precision = Tracking()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            inputs.cuda()
            # targets = torch.LongTensor(np.array(targets)).cuda()
            targets = targets.cuda(non_blocking=True)

            # compute output, loss, accuracy
            output = model(inputs)
            loss = criterion(output, targets)
            accu = accuracy(output, targets, mining_save=True, csv_wr=csv_wr, inputs=inputs, i=i)

            # update the accuracy, loss
            losses.update(loss.item(), inputs.size(0))
            precision.update(accu, inputs.size(0))

            # update the batch_time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      '|| Batch_Time: {batch_time.avg:.3f}'
                      '|| Loss:  {losses.avg:.3f}'
                      '|| accuracy: {precision.avg:.3f}'
                    .format(
                    epoch, i, len(val_loader),
                    batch_time=batch_time,
                    losses=losses,
                    precision=precision))

    logger.scalar_summary('val_loss', losses.avg, epoch)
    logger.scalar_summary('val_prec', precision.avg, epoch)
    return precision.avg


if __name__ == '__main__':
    main()
