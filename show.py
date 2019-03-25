import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import pandas as pd

from tqdm import tqdm
from utils import load_model
from expman import Experiment


def train(args):
    exps = Experiment.gather(args.run, main='model')
    exps = Experiment.filter(args.filter, exps)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for exp in exps:
        # if run.params.loc[0, 'lr'] == 0.1: continue
        try:
            exp.log.plot('epoch', 'test_acc', label=exp.path, ax=ax)
            print(exp.path)
        except Exception as e:
            print(exp.path)
            print(e)

    plt.minorticks_on()
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    plt.grid(b=True, which='minor', linewidth=0.5, linestyle='--')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(args.output, bbox_inches="tight")


def t1(args):
    exps = Experiment.gather(args.run, main='model')
    exps = Experiment.filter(args.filter, exps)

    results = Experiment.collect_all(exps, 'tradeoff.csv')

    unique_cols = results.apply(pd.Series.nunique) == 1
    common_params = results.loc[:, unique_cols].iloc[0]

    r = results.loc[:, ~unique_cols]
    metric_cols = {'t1', 'test_acc', 'test_loss', 'test_nfe'}

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))

    title = Experiment.abbreviate(common_params, main='model')
    ax1.set_title(title)
    ax1.set_ylabel('Test Accuracy')
    ax2.set_ylabel('Test NFE')
    ax2.set_xlabel('ODE final integration time $t_1$')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_xlabel('Test NFE')

    if set(r.columns) == metric_cols:  # single line plot
        r = r.sort_values('t1')
        ax1.plot(r.t1, r.test_acc, marker='.')
        ax2.plot(r.t1, r.test_nfe, marker='.')

    else:
        # print(r, metric_cols)
        grouping_cols = r.columns.difference(metric_cols).tolist()
        # print(grouping_cols)
        for name, group in r.groupby(grouping_cols):
            # print(grouping_cols)
            # print(group.reset_index())
            params = group.reset_index().loc[0, grouping_cols]
            name = Experiment.abbreviate(params, main='model')
            r = group.sort_values('t1')
            ax1.plot(r.t1, r.test_acc, label=name, marker='.')
            ax2.plot(r.t1, r.test_nfe, label=name, marker='.')
            ax3.plot(r.test_nfe, r.test_acc, label=name, marker='.')

        ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")

    plt.minorticks_on()
    for ax in (ax1, ax2):
        ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid(b=True, which='minor', linewidth=0.5, linestyle='--')
    plt.savefig(args.output, bbox_inches="tight")


def best(args):
    exps = Experiment.gather(args.run, main='model')
    exps = Experiment.filter(args.filter, exps)

    if args.l:  # search best in logs data
        results = Experiment.collect_all(exps, 'log', index=0)
    else:
        results = Experiment.collect_all(exps, 'results')

    metric_cols = {'epoch', 't1', 'test_acc', 'test_loss', 'test_nfe', 'test_tol', 'acc', 'loss'}
    grouping_cols = results.columns.difference(metric_cols).tolist()

    idx_acc_max = results.groupby(grouping_cols)['test_acc'].idxmax()
    results = results.loc[idx_acc_max]

    common_params = results.apply(pd.Series.nunique) == 1
    common_values = results.loc[:, common_params].iloc[0]
    results = results.loc[:, ~common_params]

    with pd.option_context('display.width', None), pd.option_context('max_columns', None):
        print(results.sort_values('test_acc', ascending=False).head(args.n))

    print(common_values)


def nfe(args):
    assert Experiment.is_exp_dir(args.run), "Not a run dir: args.run"
    runs = Experiment.from_dir(args.run, main='model')
    nfe = pd.read_csv(runs.path_to('nfe.csv'), index_col=0)
    sns.boxplot(x='y_true', y='nfe', data=nfe)
    plt.savefig(args.output, bbox_inches="tight")


def nparams(args):
    assert Experiment.is_exp_dir(args.run), "Not a run dir: args.run"
    run = Experiment.from_dir(args.run, main='model')
    model = load_model(run)
    print(run)

    nparams = sum(p.numel() for p in tqdm(model.parameters()) if p.requires_grad)
    print(f'N. Params: {nparams / 10 ** 6:.2g}M ({nparams})')


def clean(args):
    runs = Experiment.gather(args.run, main='model')
    empty_runs = filter(lambda run: run.log.empty, runs)
    dirs = [run.path for run in empty_runs]
    n_empty_runs = len(dirs)

    print("Empty runs found: {}".format(n_empty_runs))

    if n_empty_runs:
        print('\n'.join(dirs))
        print("Delete them? [y/N] ", end='')
        if input().lower() in ('y', 'yes'):
            for r in dirs:
                command = 'rm -rf {}'.format(r)
                print(command)
                os.system(command)


def retrieval(args):
    assert Experiment.is_exp_dir(args.run), "Not a run dir: args.run"
    run = Experiment.from_dir(args.run, main='model')

    retrieval_results_file = run.path_to('retrieval.csv')
    assert os.path.exists(retrieval_results_file), f"Retrieval results file not found: {retrieval_results_file}"

    results = pd.read_csv(retrieval_results_file)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    results.plot('t1', 'mean_ap_sym', marker='.', label='sym', ax=ax1)
    results.plot('t1', 'mean_ap_asym', marker='.', label='asym', ax=ax1)
    max_diff = (results.mean_ap_asym - results.mean_ap_sym).max()
    print(f'Asym-sym max difference: {max_diff:%}')

    tradeoff_results_file = run.path_to('tradeoff.csv')
    assert os.path.exists(tradeoff_results_file), f"Tradeoff results file not found: {tradeoff_results_file}"
    tradeoff_results = pd.read_csv(tradeoff_results_file)
    tradeoff_results = tradeoff_results[tradeoff_results.test_tol == 0.001]

    merged_results = results.merge(tradeoff_results, on='t1')

    ax2.plot(merged_results.test_nfe, merged_results.mean_ap_sym, marker='.', label='sym')
    ax2.plot(merged_results.test_nfe, merged_results.mean_ap_asym, marker='.', label='asym')

    if args.baseline:
        assert Experiment.is_exp_dir(args.baseline), "Not a run dir: args.run"
        baseline_run = Experiment.from_dir(args.baseline, main='model')
        baseline_results_file = baseline_run.path_to('retrieval.csv')
        assert os.path.exists(baseline_results_file), f"Results file for baseline not found: {baseline_results_file}"
        baseline_results = pd.read_csv(baseline_results_file)

        ax1.axhline(baseline_results.mean_ap.iloc[0], c='k', label='resnet')
        ax2.axhline(baseline_results.mean_ap.iloc[0], c='k', label='resnet')

    dataset = run.params.dataset.iloc[0].upper()
    plt.suptitle('mAP vs Feature Depth (t) - {}'.format(dataset))
    ax1.set_ylabel('mAP')
    ax1.set_xlabel('Integration time')
    ax2.set_xlabel('Test NFE')
    ax1.legend(loc='best')

    plt.savefig(args.output, bbox_inches="tight")


if __name__ == '__main__':

    def run_filter(string):
        if '=' not in string:
            raise argparse.ArgumentTypeError(
                f'Filter {string} is not in format <param1>=<value1>[, <param2>=<value2>[, ...]]')
        filters = string.split(',')
        filters = map(lambda x: x.split('='), filters)
        filters = {k: v for k, v in filters}
        return filters


    parser = argparse.ArgumentParser(description='Plot stuff')
    parser.add_argument('-f', '--filter', default={}, type=run_filter)
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('run', default='runs/')
    parser_train.add_argument('-o', '--output', default='train.pdf')
    parser_train.set_defaults(func=train)

    parser_t1 = subparsers.add_parser('t1')
    parser_t1.add_argument('run', default='runs/')
    parser_t1.add_argument('-o', '--output', default='t1.pdf')
    parser_t1.set_defaults(func=t1)

    parser_nfe = subparsers.add_parser('nfe')
    parser_nfe.add_argument('run', default='runs/')
    parser_nfe.add_argument('-o', '--output', default='nfe.pdf')
    parser_nfe.set_defaults(func=nfe)

    parser_best = subparsers.add_parser('best')
    parser_best.add_argument('run', default='runs/')
    parser_best.add_argument('-o', '--output', default='best.pdf')
    parser_best.add_argument('-n', type=int, default=10)
    parser_best.add_argument('-l', action='store_true', default=False, help='get best accuracy from log')
    parser_best.set_defaults(func=best)

    parser_nfe = subparsers.add_parser('nparams')
    parser_nfe.add_argument('run', default='runs/')
    parser_nfe.set_defaults(func=nparams)

    parser_clean = subparsers.add_parser('clean')
    parser_clean.add_argument('run', default='runs/')
    parser_clean.set_defaults(func=clean)

    parser_retrieval = subparsers.add_parser('retrieval')
    parser_retrieval.add_argument('run', default='runs/')
    parser_retrieval.add_argument('-b', '--baseline', help='run to be treated as baseline')
    parser_retrieval.add_argument('-o', '--output', default='retrieval.pdf')
    parser_retrieval.set_defaults(func=retrieval)

    args = parser.parse_args()
    args.func(args)
