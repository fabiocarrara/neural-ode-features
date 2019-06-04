import argparse
import os

import matplotlib
import torch
from matplotlib import patheffects
from torchvision import transforms
from torchvision.utils import make_grid

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

import seaborn as sns

sns.set(style='whitegrid', context='paper')

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import load_model, load_test_data
from expman import Experiment


def tradeoff(args):
    exps = Experiment.gather(args.run, main='model')
    exps = Experiment.filter(args.filter, exps)

    results = Experiment.collect_all(exps, 'nfe.csv.gz')
    results = results.sort_values('downsample')

    assert results.dataset.nunique() == 1, "This plot should be drawn with only runs on a single dataset"

    results['epsilon'] = 1 - results.t1
    results['test error'] = 100 * (results.y_pred != results.y_true)

    plt.figure()
    ax = plt.gca()

    handles = []
    labels = []
    label_map = {'residual': 'ODE-Net', 'one-shot': 'Full-ODE-Net'}

    sns.lineplot(x='epsilon', y='test error', hue='downsample', style='downsample', ci='sd', markers=('o', 'o'),
                 dashes=False, data=results, ax=ax)
    ax.set_ylim([0, 100])
    ax.set_xlabel(r'$\mathrm{\mathit{\varepsilon}}$ (time anticipation)')
    ax.set_ylabel(r'test error %')

    h, l = ax.get_legend_handles_labels()

    for hi, li in zip(h[1:], l[1:]):
        hh = Line2D([], [])
        hh.update_from(hi)
        hh.set_marker(None)
        handles.append(hh)
        labels.append(label_map[li])

    ax.get_legend().remove()

    ax2 = plt.twinx()
    sns.lineplot(x='epsilon', y='nfe', hue='downsample', style='downsample', ci='sd', markers=('X', 'X'), dashes=False,
                 data=results, ax=ax2, legend=False)
    # ax2.set_ylabel(r'number of function evaluations (NFE)')
    ax2.set_ylabel(r'NFE')
    ax2.set_ylim([0, ax2.get_ylim()[1] * .9])

    handles.extend([
        Line2D([], [], marker='o', markerfacecolor='k', markeredgecolor='w', color='k'),
        Line2D([], [], marker='X', markerfacecolor='k', markeredgecolor='w', color='k'),
    ])
    labels.extend(['error', 'nfe'])

    handler_map = {h: HandlerLine2D(marker_pad=0) for h in handles[-2:]}
    plt.legend(handles=handles, labels=labels, loc='upper center', ncol=2, handler_map=handler_map)

    plt.minorticks_on()
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    # ax2.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))

    ax.grid(b=True, which='minor', linewidth=0.5, linestyle='--')
    ax2.grid(False)
    plt.xlim(0, 1)

    plt.savefig(args.output, bbox_inches="tight")


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

    metric_cols = {'epoch', 't1', 'test_acc', 'test_loss', 'test_nfe', 'test_tol', 'acc', 'loss', 'nfe-f', 'nfe-b'}
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
    exp = Experiment.from_dir(args.run, main='model')
    nfe = pd.read_csv(exp.path_to('nfe.csv.gz'))  #, index_col=0)
    nfe = nfe[(nfe.t1 == 1) & (nfe.tol == 0.001)].reset_index(drop=True)
    nfe.nfe = (nfe.nfe - 2) / 6  # show n. steps instead of NFE

    nfe = nfe[nfe.y_true == nfe.y_pred]

    dataset = exp.params.dataset.iloc[0]
    if dataset == 'mnist':
        labels = [str(i) for i in range(10)]
    elif dataset == 'cifar10':
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    nfe.y_true = nfe.y_true.apply(lambda x: labels[x])
    nfe = nfe.sort_values('y_true')
    # g = sns.FacetGrid(nfe, col='y_true')
    # g.map(sns.kdeplot, 'nfe')
    # ax = sns.boxplot(y='y_true', x='nfe', data=nfe, orient='h')
    # ax.set_xlabel('solver steps')
    # ax.set_ylabel('image class')
    
    print('{:.2g} \pm {:.2g}'.format(nfe.nfe.mean(), nfe.nfe.std()))

    min, max = nfe.nfe.min(), nfe.nfe.max()
    values = np.arange(min, max + 1)
    plt.xticks(values)
    bins = values - .5
    plt.xlim(bins[0], bins[-1])
    counts, _, _ = plt.hist(nfe.nfe, bins=bins)
    plt.grid(b=False, axis='x')
    plt.grid(b=True, which='minor', linewidth=0.5, linestyle='--', axis='y')
    plt.gca().get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    for v, c in zip(values, counts):
        plt.text(v, c, f'{c:g}', ha='center', va='bottom')

    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    """ Images """
    sns.set_style('white')

    n = 5
    pad = 20
    side = 28 if dataset == 'mnist' else 32
    side += 2  # make_grid padding
    groups = nfe.groupby('y_true')

    test_data = load_test_data(exp)
    test_data.transform = transforms.ToTensor()

    largest_nfe = groups.nfe.nlargest(n)
    high_nfe_idxs = largest_nfe.index.get_level_values(1)
    high_nfe_images = torch.stack([test_data[i][0] for i in high_nfe_idxs])
    high_nfe_grid = make_grid(high_nfe_images, nrow=n)

    smallest_nfe = groups.nfe.nsmallest(n).reset_index().sort_values(['y_true', 'nfe'], ascending=[True, False])
    low_nfe_idxs = smallest_nfe.level_1  # nsmallest in reverse order
    low_nfe_images = torch.stack([test_data[i][0] for i in low_nfe_idxs])
    low_nfe_grid = make_grid(low_nfe_images, nrow=n)
    smallest_nfe = smallest_nfe.nfe

    grid_h = low_nfe_grid.shape[1]
    img_pad = torch.zeros((3, grid_h, pad))
    grid = torch.cat((high_nfe_grid, img_pad, low_nfe_grid), 2)

    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))  # , interpolation='nearest')
    for i, (l, s) in enumerate(zip(largest_nfe, smallest_nfe)):
        y, x = (i // n), (i % n)
        y, x = (np.array((y, x)) + (0.8, 0.75)) * side
        text = plt.text(x, y, str(int(l)), fontsize=5, ha='left', va='top', color='white')
        text.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black'), patheffects.Normal()])

        disp = side * n + pad + 6
        text = plt.text(x + disp, y, str(int(s)), fontsize=5, ha='left', va='top', color='white')
        text.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black'), patheffects.Normal()])

    ax = plt.gca()
    h, _ = ax.get_ylim()
    ticks = (np.arange(10) / 10 + 1 / (2 * 10)) * h
    plt.yticks(ticks, labels)

    plt.xticks([])
    h, htxt = -0.03, -0.08
    ax.annotate('', xy=(0, h), xycoords='axes fraction', xytext=(1, h), arrowprops=dict(arrowstyle="<->", color='k'))
    ax.annotate('high NFE', xy=(0, htxt), xycoords='axes fraction', xytext=(0, htxt))
    ax.annotate('low NFE', xy=(1, htxt), xycoords='axes fraction', xytext=(0.87, htxt))

    plt.savefig(args.output2, bbox_inches="tight")


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
    exps = Experiment.gather(args.run, main='model')
    exps = Experiment.filter(args.filter, exps)

    results_file = 'retrieval.csv' if args.data is None else 'retrieval-{}.csv'.format(args.data)
    results = Experiment.collect_all(exps, results_file)

    perc_scale = 100 if args.percentage else 1

    sym_metric = '{}_sym'.format(args.metric)
    asym_metric = '{}_asym'.format(args.metric)

    assert sym_metric in results.columns, f'Results not available for this run: {sym_metric}'
    assert asym_metric in results.columns, f'Results not available for this run: {asym_metric}'

    results[[sym_metric, asym_metric]] *= perc_scale
    results = results.sort_values('downsample')
    is_baseline = (results.downsample == 'residual') & (results.model == 'resnet')
    baseline = results[is_baseline]
    # baseline = pd.DataFrame()
    results = results[~is_baseline]

    assert results.dataset.nunique() == 1, "This plot should be drawn with only runs on a single dataset: {}".format(results.dataset.unique())

    ci = None

    plt.figure()
    ax = plt.gca()
    eps = 0
    if not baseline.empty:
        eps = .05
        common = dict(xycoords='data', textcoords='offset points', fontsize=8, va='center', ha='center')
        mean_aps = baseline.groupby('t1').mean().sort_values('t1')
        for block, aps in enumerate(mean_aps.to_dict('records')):
            if aps[sym_metric] < .95 * perc_scale:
                ax.plot([0, 1 + eps], [aps[sym_metric]]*2, c='k', lw=.8)
                ax.annotate(f'#{block}', xy=(1 + eps, aps[sym_metric]), xytext=(8, 0), **common)
                
            if aps[asym_metric] < .95 * perc_scale:
                ax.plot([-eps, 1], [aps[asym_metric]]*2, c='k', lw=.8, ls='dashed')
                ax.annotate(f'#{block}', xy=(-eps, aps[asym_metric]), xytext=(-8, 0), **common)

    sns.lineplot(x='t1', y=sym_metric, hue='downsample', style='downsample', ci=ci, markers=('o', 'o'), dashes=False,
                 data=results, ax=ax)
    sns.lineplot(x='t1', y=asym_metric, hue='downsample', style='downsample', ci=ci, markers=('o', 'o'),
                 dashes=((2, 2), (2, 2)), data=results, ax=ax)

    label_map = {'residual': 'ODE-Net', 'one-shot': 'Full-ODE-Net'}

    h, l = ax.get_legend_handles_labels()
    h_and_l = zip(h, l)
    h_and_l = sorted(h_and_l, key=lambda x: x[1], reverse=True)
    h_and_l = (
        (h, '{}, {}'.format(label_map[l], 'asymmetric' if h.is_dashed() else 'symmetric'))
        for h, l in h_and_l if l in label_map)
    h, l = zip(*h_and_l)
    
    if not baseline.empty:
        h += (Line2D([], [], c='k', lw=.8), Line2D([], [], c='k', ls='dashed', lw=.8))
        l += ('ResNet, symmetric', 'ResNet, asymmetric')

    # plt.title(dataset)
    plt.xlabel(r'$\mathrm{\mathit{t}}$ (final hidden state time)')
    plt.ylabel(r'mean Average Precision {}(%)'.format('@ 10 ' if args.metric == 'ap10' else ''))
    plt.xlim(-2*eps, 1 + 2*eps)

    y_lim = (0, perc_scale) if args.data is None else (.24 * perc_scale, .68 * perc_scale)
    plt.ylim(*y_lim)

    major_ticks = np.arange(0, 1.2, 0.2)
    minor_ticks = np.arange(0, 1.05, 0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(b=True, which='minor', linewidth=0.5, linestyle='--')

    plt.legend(handles=h, labels=l, loc='lower center', ncol=3, prop={'size': 8})
    plt.savefig(args.output, bbox_inches="tight")


def transfer(args):
    exps = Experiment.gather(args.run, main='model')
    exps = Experiment.filter(args.filter, exps)

    results_file = 'finetune.csv' if args.data is None else 'finetune-{}.csv'.format(args.data)
    results = Experiment.collect_all(exps, results_file)

    perc_scale = 100 if args.percentage else 1

    # filter out aggregations for now
    results = results[results.t1 >= 0]
    results.cv_accuracy *= perc_scale

    results['name'] = None
    results.loc[(results.downsample == 'residual') & (results.model == 'resnet'), 'name'] = 'Res-Net'
    results.loc[(results.downsample == 'residual') & (results.model == 'odenet'), 'name'] = 'Res-ODE'
    results.loc[(results.downsample == 'one-shot') & (results.model == 'odenet'), 'name'] = 'ODE-Only'
    results['name'] = pd.Categorical(results['name'], ['Res-Net', 'Res-ODE', 'ODE-Only'])
    results = results.sort_values('name')

    ax = sns.lineplot(x='t1', y='cv_accuracy', hue='name', style='name', markers=('D', 'o', 'o'), dashes=False, data=results)
    h, l = ax.get_legend_handles_labels()
    h, l = h[1:], l[1:]

    ax.lines[0].set_linestyle('--')
    h[0].set_linestyle('--')
    ax.lines[0].set_color('k')
    h[0].set_color('k')

    for hi in h:
        hi.set_markeredgecolor('w')

    plt.xlabel('t')
    plt.ylabel('5-fold Accuracy (%)')
    plt.xlim(0, 1)
    # ax.set_ylim(bottom=0)

    plt.legend(handles=h, labels=l, loc='best', ncol=1)  # , prop={'size': 8})
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

    parser_tradeoff = subparsers.add_parser('tradeoff')
    parser_tradeoff.add_argument('run', default='runs/')
    parser_tradeoff.add_argument('-o', '--output', default='tradeoff.pdf')
    parser_tradeoff.set_defaults(func=tradeoff)

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
    parser_nfe.add_argument('-o2', '--output2', default='nfe2.pdf')
    parser_nfe.set_defaults(func=nfe)

    parser_best = subparsers.add_parser('best')
    parser_best.add_argument('run', default='runs/')
    parser_best.add_argument('-o', '--output', default='best.pdf')
    parser_best.add_argument('-n', type=int, default=10)
    parser_best.add_argument('-l', action='store_true', default=False, help='get best accuracy from log')
    parser_best.set_defaults(func=best)

    parser_nparams = subparsers.add_parser('nparams')
    parser_nparams.add_argument('run', default='runs/')
    parser_nparams.set_defaults(func=nparams)

    parser_clean = subparsers.add_parser('clean')
    parser_clean.add_argument('run', default='runs/')
    parser_clean.set_defaults(func=clean)

    parser_retrieval = subparsers.add_parser('retrieval')
    parser_retrieval.add_argument('run', default='runs/')
    parser_retrieval.add_argument('-p', '--percentage', default=False, action='store_true')
    parser_retrieval.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200', 'cifar10'))
    parser_retrieval.add_argument('-m', '--metric', default='ap', choices=('ap', 'ap10'))
    parser_retrieval.add_argument('-o', '--output', default='retrieval.pdf')
    parser_retrieval.set_defaults(func=retrieval)

    parser_transfer = subparsers.add_parser('transfer')
    parser_transfer.add_argument('run', default='runs/')
    parser_transfer.add_argument('-p', '--percentage', default=False, action='store_true')
    parser_transfer.add_argument('-d', '--data', default=None, choices=('tiny-imagenet-200', 'cifar10'))
    parser_transfer.add_argument('-o', '--output', default='transfer.pdf')
    parser_transfer.set_defaults(func=transfer)

    args = parser.parse_args()
    args.func(args)
