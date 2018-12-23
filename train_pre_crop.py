import argparse
import os
import shutil
import time
import math


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
# import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import optim
from Models.densenet import densenet201, densenet121
from Models.inceptionv3 import inception_v3
from Models.inceptionv4 import inception_v4
from Utils.logger import Logger


momentum = 0.9
print_freq = 1
vali_freq = 10

testdir = '/data/ABUS_cache/Stomach_cancer/data/test'
validir_old = '/data/ABUS_cache/Stomach_cancer/data/validate'
traindir_old = '/data/ABUS_cache/Stomach_cancer/data/train'

traindir_new = '/data/ABUS_stage2/Pathology/xcz/patch/train'
validir_new = '/data/ABUS_stage2/Pathology/xcz/patch/valid'

cudnn.enabled = True
# Enables benchmark mode in cudnn, to enable the inbuilt cudnn auto-tuner
# to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='StomachCancer Training')
parser.add_argument('--learning_rate', '-lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--weight_decay', '-wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt_path', '-ckpt', default='', help='checkpoint path to load')
parser.add_argument('--ckpt_path_save', '-ckpt_s', default='ckpt/', help='checkpoint path to save')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-end', default=200, type=int, help='end epoch')
parser.add_argument('--batch_size', '-b', default=72, type=int, help='batch size')
parser.add_argument('--step_size', '-ss', default=50, type=int, help='Period of learning rate decay. ')
parser.add_argument('--log_path', '-lp', default='./log/', help='log path')
parser.add_argument('--experiment_id', '-eid', default='0', help='experiment id')
parser.add_argument('-sgd', '--use_sgd', dest='sgd', action='store_true',
                    help='Use sgd')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Just use the eval mode')
parser.add_argument('--model_n', '-mn', default='densenet121', type=str,
                    help='the model using to train')
# ====================================================================================================

args = parser.parse_args()
log_path = os.path.join(args.log_path, args.experiment_id)
logger = Logger(log_path)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.40],
                                 std=[0.229, 0.224, 0.225])

train_dataset_old = datasets.ImageFolder(
    traindir_old,
    transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, saturation=0.5, contrast=0.5, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ]))
train_dataset_new = datasets.ImageFolder(
    traindir_new,
    transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, saturation=0.5, contrast=0.5, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ]))
train_dataset = train_dataset_old + train_dataset_new
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=True, num_workers=12)

test_dataset = datasets.ImageFolder(
    testdir,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                          shuffle=False, pin_memory=True, num_workers=12)

vali_dataset_old = datasets.ImageFolder(
    validir_old,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
vali_dataset_new = datasets.ImageFolder(
    validir_new,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
vali_dataset = vali_dataset_new + vali_dataset_old
vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=args.batch_size,
                                          shuffle=False, pin_memory=True, num_workers=12)

best_accu = 0


# ====================================================================================================

def main():
    global best_accu, args

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

    if args.sgd:
        print('Using SGD')
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay,
                              momentum=0.9,
                              nesterov=False,
                              l2_reg=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    else:
        print('Using Adam')
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay,
                               amsgrad=False,
                               l2_reg=False)
        lambda1 = lambda epoch: 1 / math.sqrt(epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # use mutil process model
    model = torch.nn.DataParallel(model).cuda()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    # resume
    if args.resume:
        if os.path.isfile(args.ckpt_path):
            print("=> loading checkpoint '{}'".format(args.ckpt_path))
            checkpoint = torch.load(args.ckpt_path)
            start_epoch = checkpoint['epoch']
            best_accu = checkpoint['best_accu']
            model.load_state_dict(checkpoint['state_dict'])
            # if 'optimizer' in checkpoint:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.ckpt_path))

    # Jump over the train process
    if args.evaluate:
        validate(vali_loader, model, criterion, 0, logger)
        return

    for epoch in range(start_epoch, end_epoch):
        # adjust_learning_rate(optimizer, epoch)
        if args.sgd:
            scheduler.step()
        # train
        train(train_loader, model, criterion, optimizer, epoch, logger)

        if epoch % vali_freq == 0:
            # test evaluate
            accu = validate(vali_loader, model, criterion, epoch, logger)
        else:
            accu = validate(test_loader, model, criterion, epoch, logger)

        # remember best accuracy and save checkpoint
        is_best = accu > best_accu
        best_accu = max(accu, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Densenet121",
            'state_dict': model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
        }, is_best)


# ====================================================================================================
def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        accu = computer_accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        accuracy.update(accu, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
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
                  'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, accuracy=accuracy))
    logger.scalar_summary('train_loss', losses.avg, epoch)
    logger.scalar_summary('train_prec', accuracy.avg, epoch)


# ====================================================================================================
def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            accu = computer_accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            accuracy.update(accu, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % print_freq == 0:
            if True:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accuracy=accuracy))

        print(' * Prec@1 {accuracy.avg:.3f}'
              .format(accuracy=accuracy))
    logger.scalar_summary('test_loss', losses.avg, epoch)
    logger.scalar_summary('test_prec', accuracy.avg, epoch)
    return accuracy.avg


# ====================================================================================================
def save_checkpoint(state, is_best):
    ckpt_path_save = os.path.join(args.ckpt_path_save, args.experiment_id)
    if not os.path.exists(ckpt_path_save):
        os.mkdir(ckpt_path_save)
    path = os.path.join(ckpt_path_save, 'ckpt.pth.tar')
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(ckpt_path_save, 'model_best.pth.tar')
        shutil.copyfile(path, best_path)


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


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.learning_rate * (0.1 ** (epoch // adjust_fr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def computer_accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.max(output, 1)

        correct_k = pred.eq(target.view_as(pred)).sum().item()
        res = 100. * correct_k / batch_size

        return res


if __name__ == '__main__':
    main()
