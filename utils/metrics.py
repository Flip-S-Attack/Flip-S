import torch
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = (val * 100) / n
        self.sum += val
        self.count += n
        self.avg = (100 * self.sum) / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True).tolist()[0]
            res.append(correct_k)
        return res

def imagenet_acc(model, val_loader, device):
    print('[+] Start eval on ImageNet1k')
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for inputs, target in tqdm(val_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            # compute output
            outputs = model(inputs)
            logits = outputs.logits

            # measure accuracy and record loss
            batch_size = target.size(0)
            acc1 = accuracy(logits, target)[0]
            top1.update(acc1, batch_size)

    print('[+] Acc@1 {top1.avg:.3f}'.format(top1=top1),flush=True)
    
    return top1.avg, top5.avg

def imagenet_asr(model, val_loader, trigger, mask, normalize, target_class, device):
    print('[+] Start eval on ImageNet1k')
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for inputs, target in tqdm(val_loader):
            inputs = inputs.to(device).to(torch.float16)
            target = target.to(device) * 0 + target_class
            # compute output
            outputs = model(normalize(inputs*(1-mask) + torch.mul(trigger, mask)))
            logits = outputs.logits

            # measure accuracy and record loss
            batch_size = target.size(0)
            acc1 = accuracy(logits, target, topk=(1, ))[0]
            top1.update(acc1, batch_size)

    print('[+] Asr@1 {top1.avg:.3f}'.format(top1=top1),flush=True)
    
    return top1.avg, top5.avg
