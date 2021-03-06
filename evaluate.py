import argparse
import itertools
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm, trange

from expman import Experiment
from utils import load_test_data, load_model, TinyImageNet200


def features(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)

    params = next(run.params.itertuples())

    features_file = 'features.h5' if args.data is None else 'features-{}.h5'.format(args.data)
    features_file = run.path_to(features_file)
    dependecy_file = run.ckpt('best')

    if os.path.exists(features_file) and os.path.getctime(features_file) >= os.path.getctime(dependecy_file) and not args.force:
        print('Features file already exists, skipping...')
        sys.exit(0)

    if args.data == 'cifar10':  # using cifar10 on a tiny-imagenet-200 trained network, resize to 64 and use tiny-imagenet-200 normalization
        transfer_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        test_data = CIFAR10('data/cifar10', download=True, train=False, transform=transfer_transform)
    elif args.data == 'tiny-imagenet-200':
        transfer_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_data = TinyImageNet200('data/tiny-imagenet-200', split='val', transform=transfer_transform)
    else:
        test_data = load_test_data(run)
    test_loader = DataLoader(test_data, batch_size=params.batch_size, shuffle=False)

    model = load_model(run)
    model = model.to(args.device)
    model.eval()
    model.to_features_extractor()

    if params.model == 'odenet':
        model.odeblock.t1 = args.t1
        if 'ode' in params.downsample:
            model.downsample.odeblock.t1 = args.t1
    else:
        args.t1 = np.linspace(0, 1, 7)  # = 1 input + 6 resblocks' outputs
        args.tol = [0]

    tols = np.array(args.tol)
    t1s = np.array(args.t1)
    features = []
    y_true = []

    with torch.no_grad():
        y_true = [y.numpy() for _, y in tqdm(test_loader)]
        y_true = np.concatenate(y_true)

        for tol in tqdm(tols):
            if params.model == "odenet":
                model.odeblock.tol = tol

            f = [model(x.to(args.device)).cpu().numpy()
                 for x, _ in tqdm(test_loader)]
            f = np.concatenate(f, -2)  # concat along batch dimension

            features.append(f)

        features = np.stack(features)

    with h5py.File(features_file, 'w') as f:
        f['features'] = features
        f['y_true'] = y_true
        f['tols'] = tols
        f['t1s'] = t1s


def nfe(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)
    results_file = run.path_to('nfe.csv.gz')
    best_ckpt_file = run.ckpt('best')

    results = pd.DataFrame()
    # check if results exists and are updated, then skip the computation
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(best_ckpt_file) and not args.force:
        results = pd.read_csv(results_file, float_precision='round_trip').round({'t1': 2})

    test_data = load_test_data(run)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = load_model(run)
    model = model.to(args.device)
    model.eval()

    def _nfe(test_loader, model, t1, tol, args):
        model.odeblock.t1 = t1
        model.odeblock.tol = tol

        y_true = []
        y_pred = []
        nfes = []

        for x, y in tqdm(test_loader):
            y_true.append(y.item())
            y_pred.append(model(x.to(args.device)).argmax(dim=1).item())
            nfes.append(model.nfe(reset=True))

        return {'y_true': y_true, 'y_pred': y_pred, 'nfe': nfes}

    progress = tqdm(itertools.product(args.tol, args.t1))
    for tol, t1 in progress:
        if 't1' in results.columns and 'tol' in results.columns and ((results.t1 == t1) & (results.tol == tol)).any():
            print(f'Skipping tol={tol} t1={t1} ...')
            continue

        progress.set_postfix({'tol': tol, 't1': t1})
        result = _nfe(test_loader, model, t1, tol, args)
        result = pd.DataFrame(result)
        result['t1'] = t1
        result['tol'] = tol
        results = results.append(result, ignore_index=True)
        results.to_csv(results_file, index=False)


def tradeoff(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)
    results_file = run.path_to('tradeoff.csv')
    best_ckpt_file = run.ckpt('best')

    results = pd.DataFrame()
    # check if results exists and are updated, then load them (and probably skip the computation them later)
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(
            best_ckpt_file) and not args.force:
        results = pd.read_csv(results_file, float_precision='round_trip')

    params = next(run.params.itertuples())

    test_data = load_test_data(run)
    test_loader = DataLoader(test_data, batch_size=params.batch_size, shuffle=False)

    model = load_model(run)
    model = model.to(args.device)
    model.eval()

    def _evaluate(loader, model, t1, tol, args):
        model.odeblock.t1 = t1
        model.odeblock.tol = tol

        n_correct = 0
        n_batches = 0
        n_processed = 0
        nfe_forward = 0

        progress = tqdm(loader)
        for x, y in progress:
            x, y = x.to(args.device), y.to(args.device)
            p = model(x)
            nfe_forward += model.nfe(reset=True)
            loss = F.cross_entropy(p, y)

            n_correct += (y == p.argmax(dim=1)).sum().item()
            n_processed += y.shape[0]
            n_batches += 1

            logloss = loss.item() / n_processed
            accuracy = n_correct / n_processed
            nfe = nfe_forward / n_batches
            metrics = {
                'loss': f'{logloss:4.3f}',
                'acc': f'{n_correct:4d}/{n_processed:4d} ({accuracy:.2%})',
                'nfe': f'{nfe:3.1f}'
            }
            progress.set_postfix(metrics)

        metrics = {'t1': t1, 'test_loss': logloss, 'test_acc': accuracy, 'test_nfe': nfe, 'test_tol': tol}
        return metrics

    progress = tqdm(itertools.product(args.tol, args.t1))
    for tol, t1 in progress:
        if 't1' in results.columns and 'test_tol' in results.columns and ((results.t1 == t1) & (results.test_tol == tol)).any():
            print(f'Skipping tol={tol} t1={t1} ...')
            continue

        progress.set_postfix({'tol': tol, 't1': t1})
        result = _evaluate(test_loader, model, t1, tol, args)
        results = results.append(result, ignore_index=True)
        results.to_csv(results_file, index=False)


def accuracy(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)
    results_file = run.path_to('results')
    best_ckpt_file = run.ckpt('best')

    all_results = pd.DataFrame()
    # check if results exists and are updated, then load them (and probably skip the computation them later)
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(
            best_ckpt_file) and not args.force:
        all_results = pd.read_csv(results_file, float_precision='round_trip')

    params = next(run.params.itertuples())

    test_data = load_test_data(run)
    test_loader = DataLoader(test_data, batch_size=params.batch_size, shuffle=False)

    model = load_model(run)
    model = model.to(args.device)
    model.eval()

    t1 = torch.arange(0, 1.05, .05)  # from 0 to 1 w/ .05 step
    model.odeblock.t1 = t1[1:]  # 0 is implicit
    model.odeblock.return_last_only = False

    if params.downsample == 'ode2':
        model.downsample.odeblock.t1 = t1[1:]  # 0 is implicit
        model.downsample.odeblock.return_last_only = False
        model.downsample.odeblock.apply_conv = True
        t1 = torch.cat((t1, t1))

    T = len(t1)

    def _evaluate(loader, model, tol, args):
        model.odeblock.tol = tol
        if 'ode' in params.downsample:
            model.downsample.odeblock.tol = tol

        n_batches = 0
        n_processed = 0
        nfe_forward = 0

        n_correct = torch.zeros(T)
        tot_losses = torch.zeros(T)

        progress = tqdm(loader)
        for x, y in progress:
            x, y = x.to(args.device), y.to(args.device)
            p = model(x)  # timestamps (T) x batch (N) x classes (C)
            nfe_forward += model.nfe(reset=True)
            pp = p.permute(1, 2, 0)  # N x C x T
            yy = y.unsqueeze(1).expand(-1, T)  # N x T
            losses = F.cross_entropy(pp, yy, reduction='none')  # N x T

            tot_losses += losses.sum(0).cpu()

            yy = y.unsqueeze(0).expand(T, -1)
            n_correct += (yy == p.argmax(dim=-1)).sum(-1).float().cpu()
            n_processed += y.shape[0]
            n_batches += 1

            # logloss = losses.item() / n_processed
            # accuracy = n_correct / n_processed
            nfe = nfe_forward / n_batches
            metrics = {
                # 'loss': f'{logloss:4.3f}',
                # 'acc': f'{n_correct:4d}/{n_processed:4d} ({accuracy:.2%})',
                'nfe': f'{nfe:3.1f}'
            }
            progress.set_postfix(metrics)

        loglosses = tot_losses / n_processed
        accuracies = n_correct / n_processed

        metrics = {'t1': t1.numpy(),
                   'test_loss': loglosses.numpy(),
                   'test_acc': accuracies.numpy(),
                   'test_nfe': [nfe, ] * T,
                   'test_tol': [tol, ] * T}

        return metrics

    progress = tqdm(args.tol)
    with torch.no_grad():
        for tol in progress:
            progress.set_postfix({'tol': tol})

            if 'test_tol' in all_results.columns and (all_results.test_tol == tol).any():
                progress.write(f'Skipping: tol={tol:g}')
                continue

            results = _evaluate(test_loader, model, tol, args)
            results = pd.DataFrame(results)
            all_results = all_results.append(results, ignore_index=True)
            all_results.to_csv(results_file, index=False)


def retrieval(args):
    exp = Experiment.from_dir(args.run, main='model')

    features_file = 'features.h5' if args.data is None else 'features-{}.h5'.format(args.data)
    results_file = 'retrieval.csv' if args.data is None else 'retrieval-{}.csv'.format(args.data)

    features_file = exp.path_to(features_file)
    results_file = exp.path_to(results_file)

    assert os.path.exists(features_file), f"No pre-extracted features found: {features_file}"

    all_results = pd.DataFrame()
    # check if results exists and are updated, then load them (and probably skip the computation them later)
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(features_file) and not args.force:
        all_results = pd.read_csv(results_file, float_precision='round_trip')

    with h5py.File(features_file, 'r') as f:
        features = f['features'][...]
        y_true = f['y_true'][...]
        t1s = f['t1s'][...]

    features /= np.linalg.norm(features, axis=-2, keepdims=True) + 1e-7

    queries = features  # all queries

    n_samples = features.shape[-2]  # number of samples, for both models (first dimension might be t1)
    n_queries = queries.shape[-2]  # number of queries, for both models (first dimension might be t1)

    gt = np.broadcast_to(y_true, (n_queries, n_samples)) == y_true[:n_queries].reshape(n_samples, -1)  # gt per query in each row

    def score(queries, db, gt, k=None):
        scores = queries.dot(db.T)
        if k is None:  # average precision
            aps = [average_precision_score(gt[i], scores[i]) for i in trange(n_queries)]
        else:  # average precision at k
            ranking = scores.argsort(axis=1)[:, ::-1][:, :k]  # top k indexes for each query
            ranked_scores = scores[np.arange(n_queries)[:, np.newaxis], ranking]
            ranked_gt = gt[np.arange(n_queries)[:, np.newaxis], ranking]
            aps = [average_precision_score(ranked_gt[i], ranked_scores[i]) for i in trange(n_queries)]  # avg. prec. @ k

        return aps

    for i, t1 in enumerate(tqdm(t1s)):
        # TODO check and skip
        ap_asym = score(queries[i], features[-1], gt)  # t1 = 1 for db
        ap_sym = score(queries[i], features[i], gt)  # t1 same for queries and db

        ap10_asym = score(queries[i], features[-1], gt, k=10)
        ap10_sym = score(queries[i], features[i], gt, k=10)

        results = pd.DataFrame({'ap_asym': ap_asym, 'ap_sym': ap_sym, 'ap10_asym': ap10_asym, 'ap10_sym': ap10_sym})
        results['t1'] = t1
        all_results = all_results.append(results, ignore_index=True)
        all_results.to_csv(results_file, index=False)


def finetune(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)

    features_file = 'features.h5' if args.data is None else 'features-{}.h5'.format(args.data)
    features_file = run.path_to(features_file)
    results_file = 'finetune.csv' if args.data is None else 'finetune-{}.csv'.format(args.data)
    results_file = run.path_to(results_file)

    assert os.path.exists(features_file), f"Features file not found: {features_file}"

    results = pd.DataFrame()
    if os.path.exists(results_file):
        if os.path.getctime(results_file) >= os.path.getctime(features_file) and not args.force:
            results = pd.read_csv(results_file)

    params = next(run.params.itertuples())

    with h5py.File(features_file, 'r') as f:
        features = f['features'][...]
        y_true = f['y_true'][...]
        t1s = f['t1s'][...]

    if args.aggregate:
        features = features.mean(0, keepdims=True)
        t1s = np.array([-1])

    block = np.zeros_like(t1s, dtype=int)
    if params.downsample == "ode":
        block = np.concatenate((block, block + 1))
        t1s = np.concatenate((t1s, t1s))

    svm_dir = run.path_to('svms/')
    os.makedirs(svm_dir, exist_ok=True)

    svm = LinearSVC()
    Cs = np.logspace(-2, 2, 5)
    svm = GridSearchCV(svm, {'C': Cs}, scoring='accuracy', n_jobs=-1, verbose=2, cv=5)

    for t1, b, fi in tqdm(zip(t1s, block, features)):
        if 't1' in results.columns and 'block' in results.columns and ((results.t1 == t1) & (results.block == b)).any():
            print(f'Skipping b={b} t1={t1} ...')
            continue

        score = svm.fit(fi, y_true).best_score_
        print(f'Accuracy: {score:.2%}')
        results = results.append({'block': b, 't1': t1, 'cv_accuracy': score}, ignore_index=True)
        results.to_csv(results_file, index=False)
        svm_file = run.path_to(f'svms/svm_b{b}_t{t1}.pkl')
        joblib.dump(svm, svm_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ODENet/ResNet evaluations')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--force', action='store_true', help='Force evaluation computation')
    parser.set_defaults(cuda=True, force=False)
    subparsers = parser.add_subparsers()

    default_tol = (0.001, 0.01, 0.1, 1, 10, 100)
    default_t1 = tuple(np.linspace(0, 1, 21).round(2))

    parser_tradeoff = subparsers.add_parser('tradeoff')
    parser_tradeoff.add_argument('run')
    parser_tradeoff.add_argument('--tol', type=float, nargs='+', default=default_tol)
    parser_tradeoff.add_argument('--t1', type=float, nargs='+', default=default_t1)
    parser_tradeoff.set_defaults(func=tradeoff)

    parser_accuracy = subparsers.add_parser('accuracy')
    parser_accuracy.add_argument('run')
    parser_accuracy.add_argument('-t', '--tol', type=float, nargs='+', default=default_tol)
    parser_accuracy.set_defaults(func=accuracy)

    parser_nfe = subparsers.add_parser('nfe')
    parser_nfe.add_argument('run')
    parser_nfe.add_argument('--tol', type=float, nargs='+', default=default_tol)
    parser_nfe.add_argument('--t1', type=float, nargs='+', default=default_t1)
    parser_nfe.set_defaults(func=nfe)

    parser_features = subparsers.add_parser('features')
    parser_features.add_argument('run')
    parser_features.set_defaults(func=features)
    parser_features.add_argument('--t1', type=float, nargs='+', default=default_t1)
    parser_features.add_argument('--tol', type=float, nargs='+', default=default_tol)
    parser_features.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200', 'cifar10'))

    parser_retrieval = subparsers.add_parser('retrieval')
    parser_retrieval.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200', 'cifar10'))
    parser_retrieval.add_argument('run')
    parser_retrieval.set_defaults(func=retrieval)

    parser_finetune = subparsers.add_parser('finetune')
    parser_finetune.add_argument('run')
    parser_finetune.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200',))
    parser_finetune.add_argument('-a', '--aggregate', default=False, action='store_true')
    parser_finetune.set_defaults(func=finetune)
    args = parser.parse_args()

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    args.func(args)
