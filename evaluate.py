import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import itertools

from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
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
    results_file = run.path_to('results')
    dependecy_file = run.ckpt('best')

    if os.path.exists(features_file) and os.path.getctime(features_file) >= os.path.getctime(dependecy_file) and not args.force:
        print('Skipping...')
        sys.exit(0)

    if args.data == 'tiny-imagenet-200':
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
        if os.path.exists(results_file):  # reuse t1s if already tested
            results = pd.read_csv(results_file)
            results = results[results.t1 <= 1]
            t1s = results.t1.sort_values().unique()
        else:
            t1s = np.arange(.05, 1.05, .05)  # from 0 to 1 w/ .05 step

        model.odeblock.t1 = t1s.tolist()
        if 'ode' in params.downsample:
            model.downsample.odeblock.t1 = t1s.tolist()

        t1s = np.insert(t1s, 0, 0)  # add 0 at the beginning

    features = []
    y_true = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(args.device)
            y_true.append(y.numpy())

            f = model(x)
            f = f.cpu().numpy()

            features.append(f)

    features = np.concatenate(features, -2)  # concat along batch dimension
    y_true = np.concatenate(y_true)

    with h5py.File(features_file, 'w') as f:
        f['features'] = features
        f['y_true'] = y_true
        if params.model == 'odenet':
            f['t1s'] = t1s


def nfe(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)
    results_file = run.path_to('nfe.csv')
    best_ckpt_file = run.ckpt('best')

    # check if results exists and are updated, then skip the computation
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(
            best_ckpt_file) and not args.force:
        print('Skipping...')
        return

    test_data = load_test_data(run)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = load_model(run)
    model = model.to(args.device)
    model.eval()
    model.odeblock.tol = args.tol

    def process(datum):
        x, y = datum
        x = x.to(args.device)
        p = model(x)
        nfe = model.nfe(reset=True)
        pred = p.argmax(dim=1).item()
        y = y.item()
        return {'y_true': y, 'y_pred': pred, 'nfe': nfe}

    data = [process(d) for d in tqdm(test_loader)]
    pd.DataFrame(data).to_csv(results_file)


def tradeoff(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)
    results_file = run.path_to('tradeoff.csv')
    best_ckpt_file = run.ckpt('best')

    results = pd.DataFrame()
    # check if results exists and are updated, then load them (and probably skip the computation them later)
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(
            best_ckpt_file) and not args.force:
        results = pd.read_csv(results_file)

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
        all_results = pd.read_csv(results_file)

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
        model.downsample.odeblock.t1 = t1[1:] # 0 is implicit
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
    features_file = exp.path_to('features.h5')
    results_file = exp.path_to('retrieval.csv')

    assert os.path.exists(features_file), f"No pre-extracted features found: {features_file}"

    all_results = pd.DataFrame()
    # check if results exists and are updated, then load them (and probably skip the computation them later)
    if os.path.exists(results_file) and os.path.getctime(results_file) >= os.path.getctime(
            features_file) and not args.force:
        all_results = pd.read_csv(results_file)

    params = next(exp.params.itertuples())

    with h5py.File(features_file, 'r') as f:
        features = f['features'][...]
        y_true = f['y_true'][...]
        if params.model == 'odenet':
            t1s = f['t1s'][...]

    features /= np.linalg.norm(features, axis=-2, keepdims=True)

    queries = features  # all queries

    n_samples = features.shape[-2]  # number of samples, for both models (first dimension might be t1)
    n_queries = queries.shape[-2]  # number of queries, for both models (first dimension might be t1)

    gt = np.broadcast_to(y_true, (n_queries, n_samples)) == y_true[:n_queries].reshape(n_samples,
                                                                                       -1)  # gt per query in each row

    def score(queries, db, gt):
        scores = queries.dot(db.T)
        aps = [average_precision_score(gt[i], scores[i]) for i in trange(n_queries)]
        return np.mean(aps)

    if params.model == 'odenet':
        for i, t1 in enumerate(tqdm(t1s)):
            # TODO check and skip
            mean_ap_asym = score(queries[i], features[-1], gt)  # t1 = 1 for db
            mean_ap_sym = score(queries[i], features[i], gt)  # t1 same for queries and db
            results = {'t1': t1, 'mean_ap_asym': mean_ap_asym, 'mean_ap_sym': mean_ap_sym}
            all_results = all_results.append(results, ignore_index=True)
            all_results.to_csv(results_file, index=False)
    else:  # resnet
        mean_ap = score(features, features, gt)
        all_results = all_results.append({'mean_ap': mean_ap}, ignore_index=True)
        all_results.to_csv(results_file, index=False)


def finetune(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)

    features_file = 'features.h5' if args.data is None else 'features-{}.h5'.format(args.data)
    features_file = run.path_to(features_file)

    assert os.path.exists(features_file), f"Features file not found: {features_file}"

    results = pd.DataFrame()

    results_file = run.path_to('finetune.csv')
    if os.path.exists(results_file):
        if os.path.getctime(results_file) >= os.path.getctime(features_file) and not args.force:
            results = pd.read_csv(results_file)

    params = next(run.params.itertuples())

    with h5py.File(features_file, 'r') as f:
        features = f['features'][...]
        y_true = f['y_true'][...]
        t1s = f['t1s'][...]

    block = np.zeros_like(t1s, dtype=int)
    if params.downsample == "ode":
        block = np.concatenate((block, block + 1))
        t1s = np.concatenate((t1s, t1s))

    svm = LinearSVC()
    Cs = np.logspace(-2, 2, 5)
    svm = GridSearchCV(svm, {'C': Cs}, scoring='accuracy', n_jobs=-1, verbose=10, cv=5)

    for t1, b, fi in tqdm(zip(t1s, block, features)):
        if 't1' in results.columns and 'block' in results.columns and ((results.t1 == t1) & (results.block == b)).any():
            print(f'Skipping b={b} t1={t1} ...')
            continue

        score = svm.fit(fi, y_true).best_score_
        print(f'Accuracy: {score:.2%}')
        results = results.append({'block': b, 't1': t1, 'cv_accuracy': score}, ignore_index=True)
        results.to_csv(results_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ODENet/ResNet evaluations')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--force', action='store_true', help='Force evaluation computation')
    parser.set_defaults(cuda=True, force=False)
    subparsers = parser.add_subparsers()

    parser_tradeoff = subparsers.add_parser('tradeoff')
    parser_tradeoff.add_argument('run')
    parser_tradeoff.add_argument('--tol', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1, 10, 100])
    parser_tradeoff.add_argument('--t1', type=float, nargs='+', default=np.arange(0.05, 1.05, .05).tolist())
    parser_tradeoff.set_defaults(func=tradeoff)

    parser_accuracy = subparsers.add_parser('accuracy')
    parser_accuracy.add_argument('run')
    parser_accuracy.add_argument('-t', '--tol', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1, 10, 100])
    parser_accuracy.set_defaults(func=accuracy)

    parser_nfe = subparsers.add_parser('nfe')
    parser_nfe.add_argument('run')
    parser_nfe.add_argument('-t', '--tol', type=float, default=1e-3)
    parser_nfe.set_defaults(func=nfe)

    parser_features = subparsers.add_parser('features')
    parser_features.add_argument('run')
    parser_features.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200',))
    parser_features.set_defaults(func=features)

    parser_retrieval = subparsers.add_parser('retrieval')
    parser_retrieval.add_argument('run')
    parser_retrieval.set_defaults(func=retrieval)

    parser_finetune = subparsers.add_parser('finetune')
    parser_finetune.add_argument('run')
    parser_finetune.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200',))
    parser_finetune.set_defaults(func=finetune)
    args = parser.parse_args()

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    args.func(args)
