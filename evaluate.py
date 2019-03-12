import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from expman import Experiment
from utils import load_test_data, load_model


def features(args):
    run = Experiment.from_dir(args.run, main='model')
    print(run)
    params = next(run.params.itertuples())

    features_file = run.path_to('features.h5')
    results_file = run.path_to('results')
    best_ckpt = run.ckpt('best')

    dependecy_file = best_ckpt if params.model == 'resnet' else results_file

    if os.path.exists(features_file) and os.path.getctime(features_file) >= os.path.getctime(
            dependecy_file) and not args.force:
        print('Skipping...')
        sys.exit(0)

    else:
        results_file = run.path_to('results')

    model = load_model(run)
    model = model.to(args.device)
    model.eval()
    model.to_features_extractor()

    if params.model == 'odenet':
        assert os.path.exists(results_file), 'Results file for this run not found: {}'.format(results_file)

        results = pd.read_csv(results_file)
        results = results[results.t1 <= 1]
        t1s = results.t1.sort_values().unique()
        model.odeblock.t1 = t1s.tolist()
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


def evaluate(loader, model, t1, tol, args):
    model.eval()
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


def test(args):
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

    tol_progress = tqdm(args.tol)
    for tol in tol_progress:
        tol_progress.set_postfix({'tol': tol})

        if 'test_tol' in all_results.columns:
            t1_results = all_results[all_results.test_tol == tol].sort_values('t1').reset_index(drop=True)
        else:
            t1_results = pd.DataFrame()

        n_evals = len(t1_results)
        progress = tqdm(initial=n_evals, total=args.num_evals)
        while n_evals < args.num_evals:
            if n_evals == 0:
                t1 = 1e-6
            elif n_evals == 1:
                t1 = 1
            else:
                ts = t1_results['t1'].values
                accs = t1_results['test_acc'].values
                metric = (np.diff(ts) * np.diff(accs) / accs[:-1])  # search the interval with maximum (Dt * relative accuracy increment)
                left_t_idx = metric.argmax()
                t1 = (ts[left_t_idx] + ts[left_t_idx + 1]) / 2

            progress.set_postfix({'t1': t1})
            if len(all_results) == 0 or not ((all_results.t1 == t1) & (all_results.test_tol == tol)).any():
                result = evaluate(test_loader, model, t1, tol, args)
                t1_results = t1_results.append(result, ignore_index=True).sort_values('t1').reset_index(drop=True)
                all_results = all_results.append(result, ignore_index=True)
                all_results.to_csv(run.path_to('results'), index=False)
            else:
                progress.write('Skipping: t1={:g}, tol={:g}'.format(t1, tol))

            progress.update(1)
            n_evals += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ODENet/ResNet evaluations')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--force', action='store_true', help='Force evaluation computation')
    parser.set_defaults(cuda=True, force=False)
    subparsers = parser.add_subparsers()

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('run')
    parser_test.add_argument('-t', '--tol', type=float, nargs='+', default=[0.001, 0.01, 0.1, 1, 10, 100])
    parser_test.add_argument('-n', '--num-evals', type=int, default=15)
    parser_test.set_defaults(func=test)

    parser_nfe = subparsers.add_parser('nfe')
    parser_nfe.add_argument('run')
    parser_nfe.add_argument('-t', '--tol', type=float, default=1e-3)
    parser_nfe.set_defaults(func=nfe)

    parser_features = subparsers.add_parser('features')
    parser_features.add_argument('run')
    parser_features.set_defaults(func=features)
    args = parser.parse_args()

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    args.func(args)
