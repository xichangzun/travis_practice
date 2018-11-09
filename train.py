import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np

import csv

from Utils.logger import Logger
from Models.densenet import densenet121, densenet201
from Models.inceptionv3 import inception_v3
from Models.inceptionv4 import inception_v4
from Data.data_loader import get_dataset

epochs = 1000
batch_size = 82
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-2
print_freq = 1
eval_freq = 5
adjust_fr = 200
mining_switch = False
sgd_switch = False

# ============================================================================================
# ============================================================================================
print('loading data')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomRotation(10),
                                      # transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(brightness=0.1, saturation=0.5, contrast=0.5, hue=0.05),
                                      transforms.ToTensor(),
                                      normalize,
                                      ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     normalize,
                                     ])

train_dataset, test_dataset = get_dataset(train_transform, test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=12,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=12,
                                          pin_memory=True)

best_prec1 = 0

# ============================================================================================
# ============================================================================================
# ============================================================================================

parser = argparse.ArgumentParser(description='StomachCancer Training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Just use the eval mode')
parser.add_argument('--resume', default='', type=str, metavar='path',
                    help='path to latest checkpoint')
parser.add_argument('--log', default='log/0', type=str, metavar='logpath',
                    help='the number of each test log')
parser.add_argument('--model_n', default='densenet121', type=str,
                    help='the model using to train')


def main():
    global best_prec1, args
    args = parser.parse_args()
    logger = Logger(args.log)

    if args.model_n == "densenet201":
        # select models : densenet201
        print("Using the Pre-train model {densenet201}")
        # model = models.__dict__['densenet201'](pretrained = True)
        model = densenet201(pretrained=True, drop_rate=0.1, num_classes=2)
    elif args.model_n == "densenet121":
        # select models : densenet121
        print("Using the Pre-train model {densenet121}")
        model = densenet121(pretrained=True, drop_rate=0.1, num_classes=2)
    elif args.model_n == "inceptionv3":
        # select models : inceptionv3
        print("Using the Pre-train model {inceptionv3}")
        model = inception_v3(pretrained=True, drop_rate=0.1, num_classes=2)
    else:
        # select models : inceptionv4
        print("Using the Pre-train model {inceptionv4}")
        model = inception_v4(pretrained=True, drop_rate=0.1, num_classes=2)

    # use single process model
    model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

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

    cudnn.benchmark = True

    start_epoch = 0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Jump over the train process
    if args.evaluate:
        validate(test_loader, model, criterion)
        return

    for epoch in range(start_epoch, epochs):
        if sgd_switch:
            adjust_learning_rate(optimizer, epoch)
        csv_file = open('mining_csv.csv', 'w', encoding='utf-8')
        csv_wr = csv.writer(csv_file)

        # train
        print('\n===>training with epoch ' + str(epoch) + '\n')
        train(train_loader, model, criterion, optimizer, epoch, logger, csv_wr)

        # hard_mining
        if mining_switch:
            mining_dataset = get_dataset(train_transform, test_transform, mining_mode=True)
            mining_loader = torch.utils.data.DataLoader(mining_dataset, batch_size=batch_size,
                                                        shuffle=True, pin_memory=True)
            print('\n===>mining with epoch ' + str(epoch) + '\n')
            mining(mining_loader, model, criterion, optimizer, epoch, logger)
        csv_file.close()
        if epoch % eval_freq == 0:
            # test evaluate
            print('\n===>validating with epoch ' + str(epoch) + '\n')
            prec1 = validate(test_loader, model, criterion, epoch, logger)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "Densenet121",
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger, csv_wr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    s = time.time()
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        target = torch.LongTensor(np.array(target).astype(float)).cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1 = accuracy(output, target)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    print("This Epoch using time is:", time.time()-s)
    logger.scalar_summary('train_loss', losses.avg, epoch)
    logger.scalar_summary('train_prec', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda()
            target = torch.LongTensor(np.array(target).astype(float)).cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                # if True:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
    logger.scalar_summary('test_loss', losses.avg, epoch)
    logger.scalar_summary('test_prec', top1.avg, epoch)
    return top1.avg


def mining(mining_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target, filename) in enumerate(mining_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        target = torch.LongTensor(np.array(target).astype(float)).cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                .format(
                epoch, i, len(mining_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    logger.scalar_summary('mining_loss', losses.avg, epoch)
    logger.scalar_summary('mining_prec', top1.avg, epoch)


def save_checkpoint(state, is_best, filename='./ckpt/checkpoint9.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './ckpt/model_best9.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // adjust_fr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, filename=None, mining_save=False, csv_wr=None):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():

        batch_size = target.size(0)
        _, pred = torch.max(output, 1)
        # _, pred = output.topk(1, 1, True, True)
        # pred = pred.t()

        correct = pred.eq(target.view(-1).expand_as(pred))
        correct_k = pred.eq(target.view_as(pred)).sum().item()
        # correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        # res = [correct_k.mul_(100.0 / batch_size)]
        res = 100. * correct_k / batch_size

        # saving the hard mining:
        if mining_save:
            file_list = filename
            correct = torch.squeeze(correct)
            for i in range(len(correct)):
                if correct[i] == 0:
                    target = target.view(1, -1)
                    tar = target[0][i].cpu()
                    tar = tar.numpy()
                    csv_wr.writerow([file_list[i], tar])

        return res


if __name__ == '__main__':
    main()



