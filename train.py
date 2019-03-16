import argparse
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from expman import Experiment
from model import ResNet, ODENet
from utils import load_dataset


def save_checkpoint(exp, state, is_best):
    filename = exp.ckpt('last')
    torch.save(state, filename)
    if is_best:
        best_filename = exp.ckpt('best')
        shutil.copyfile(filename, best_filename)


def train(loader, model, optimizer, args):
    model.train()
    optimizer.zero_grad()

    nfe_forward = 0
    nfe_backward = 0

    n_correct = 0
    n_processed = 0
    n_batch_processed = 0

    total_loss = 0

    progress = tqdm(loader)
    for x, y in progress:
        x, y = x.to(args.device), y.to(args.device)
        p = model(x)
        loss = F.cross_entropy(p, y)
        total_loss += loss.item()

        n_correct += (y == p.argmax(dim=1)).sum().item()
        n_processed += y.shape[0]

        nfe_forward += model.nfe(reset=True)

        loss.backward()

        nfe_backward += model.nfe(reset=True)
        n_batch_processed += 1

        if n_batch_processed % args.batch_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        accuracy = n_correct / n_processed
        avg_loss = total_loss / n_batch_processed
        avg_nfe_forward = nfe_forward / n_batch_processed
        avg_nfe_backward = nfe_backward / n_batch_processed

        progress.set_postfix({
            'loss': f'{loss:4.3f}|{avg_loss:4.3f}',
            'acc': f'{n_correct:4d}/{n_processed:4d} ({accuracy:.2%})',
            'NFE-F': f'{avg_nfe_forward:3.1f}',
            'NFE-B': f'{avg_nfe_backward:3.1f}'
        })

    return {'loss': avg_loss, 'acc': accuracy, 'nfe-f': avg_nfe_forward, 'nfe-b': avg_nfe_backward}


def evaluate(loader, model, args):
    model.eval()

    nfe_forward = 0

    n_correct = 0
    n_batches = 0
    n_processed = 0

    total_loss = 0

    progress = tqdm(loader)
    for x, y in progress:
        x, y = x.to(args.device), y.to(args.device)
        p = model(x)
        nfe_forward += model.nfe(reset=True)
        loss = F.cross_entropy(p, y, reduction='sum')
        total_loss += loss.item()

        n_correct += (y == p.argmax(dim=1)).sum().item()
        n_processed += y.shape[0]
        n_batches += 1

        logloss = total_loss / n_processed
        accuracy = n_correct / n_processed
        nfe = nfe_forward / n_batches
        metrics = {
            'loss': f'{logloss:4.3f}',
            'acc': f'{n_correct:4d}/{n_processed:4d} ({accuracy:.2%})',
            'nfe': f'{nfe:3.1f}'
        }
        progress.set_postfix(metrics)

    return {'test_loss': logloss, 'test_acc': accuracy, 'test_nfe': nfe}


def main(args):
    root = 'runs_' + args.dataset
    exp = Experiment(args, root=root, main='model', ignore=('cuda', 'device', 'epochs', 'resume'))

    print(exp)
    if os.path.exists(exp.path_to('log')) and not args.resume:
        print('Skipping ...')
        sys.exit(0)

    train_data, test_data, in_ch = load_dataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.model == 'odenet':
        model = ODENet(in_ch, n_filters=args.filters, downsample=args.downsample, tol=args.tol, adjoint=args.adjoint,
                       dropout=args.dropout)
    else:
        model = ResNet(in_ch, n_filters=args.filters, downsample=args.downsample, dropout=args.dropout)

    model = model.to(args.device)
    if args.optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # print(train_data)
    # print(test_data)
    # print(model)
    # print(optimizer)

    if args.resume:
        ckpt = torch.load(exp.ckpt('last'))
        print('Loaded: {}'.format(exp.ckpt('last')))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        start_epoch = ckpt['epoch'] + 1
        best_accuracy = exp.log['test_acc'].max()
        print('Resuming from epoch {}: {}'.format(start_epoch, exp.name))
    else:
        metrics = evaluate(test_loader, model, args)
        best_accuracy = metrics['test_acc']
        start_epoch = 1

    if args.lrschedule == 'fixed':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)  # no-op scheduler, just for cleaner code
    elif args.lrschedule == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience)
    elif args.lrschedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.lrcycle, last_epoch=start_epoch - 2)

    progress = trange(start_epoch, args.epochs + 1, initial=start_epoch, total=args.epochs)
    for epoch in progress:
        metrics = {'epoch': epoch}

        progress.set_postfix({'Best ACC': f'{best_accuracy:.2%}'})
        progress.set_description('TRAIN')
        train_metrics = train(train_loader, model, optimizer, args)

        progress.set_description('EVAL')
        test_metrics = evaluate(test_loader, model, args)

        is_best = test_metrics['test_acc'] > best_accuracy
        best_accuracy = max(test_metrics['test_acc'], best_accuracy)

        metrics.update(train_metrics)
        metrics.update(test_metrics)

        save_checkpoint(exp, {
            'epoch': epoch,
            'params': vars(args),
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'metrics': metrics
        }, is_best)

        exp.push_log(metrics)
        sched_args = metrics['test_acc'] if args.lrschedule == 'plateau' else None
        scheduler.step(sched_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ODENet/ResNet training')
    parser.add_argument('--dataset', type=str, choices=('mnist', 'cifar10', 'tiny-imagenet-200'), default='mnist')
    parser.add_argument('--augmentation', type=str, choices=('none', 'crop+flip+norm', 'crop+jitter+flip+norm'),
                        default='none')
    parser.add_argument('-m', '--model', type=str, choices=('resnet', 'odenet'), default='odenet')
    parser.add_argument('-d', '--downsample', type=str, choices=('ode', 'residual', 'convolution', 'minimal'),
                        default='residual')
    parser.add_argument('-f', '--filters', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--batch-accumulation', type=int, default=1)
    parser.add_argument('-o', '--optim', type=str, choices=('sgd', 'adam'), default='adam')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrschedule', type=str, choices=('fixed', 'plateau', 'cosine'), default='fixed')
    parser.add_argument('--lrcycle', type=int, default=0)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('--wd', type=float, default=0, help='weight decay')

    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    parser.add_argument('-t', '--tol', type=float, default=1e-3)
    parser.add_argument('-a', '--adjoint', default=False, action='store_true')

    parser.add_argument('-r', '--resume', action='store_true', default=False)
    parser.add_argument('-s', '--seed', type=int, default=23)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    main(args)
