import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from expman import Experiment, exp_filter

def figsize(cols):
    w = 3.39 if cols == 1 else 6.9  # width in inches
    gr = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
    h = w * gr # height in inches
    return w, h
       
sns.set(style='whitegrid', context='paper', font='serif', font_scale=3, rc={
    'backend': 'ps',
    'text.latex.preamble': [r'\usepackage{gensymb,amsmath}'],
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    #'text.fontsize': 8, # was 10
    'legend.fontsize': 8, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'text.usetex': True,
    #'legend.frameon': True,
    #'lines.linewidth': 2.5
})

def accuracy_vs_robustness(args):
    models = ('resnet', 'mixed', 'fullode')

    def _get_results(model):
        exp_dir = os.path.join(args.root, f'{args.dataset}_{model}', 'adv-attack')
        exps = Experiment.gather(exp_dir)
        exps = Experiment.filter(args.filter, exps)
        results = Experiment.collect_all(exps, 'results.csv')
        results['model'] = model
        return results

    results = map(_get_results, models)
    results = pd.concat(results, axis=0)

    results = results.rename({'distance_x': 'p', 'distance_y': 'distance'}, axis=1)
    results['successrate'] = np.isfinite(results.distance.values)

    natural_errors = results.distance == 0
    results = results[~natural_errors]  # discard natural errors
    
    good_eps = (0.05, 0.1, 0.3) if args.dataset == 'mnist' else (0.01, 0.03, 0.05)
    results = results[results.epsilon.isin(good_eps)]
    
    results = results.groupby(['tol','model','epsilon','p']).successrate.mean()
    results = results.reset_index()

    # lazyness ..
    mnist_acc = [[0.0001, .996, .995, .995], 
                 [0.0010, None, .995, .995], 
                 [0.0100, None, .995, .994], 
                 [0.1000, None, .995, .992], 
                 [1.0000, None, .995, .988], 
                 [10.000, None, .995, .985]]
                  
    cifar10_acc = [[0.0001, .927, .922, .909],
                   [0.0010, None, .922, .908],
                   [0.0100, None, .921, .907],
                   [0.1000, None, .921, .894],
                   [1.0000, None, .921, .887],
                   [10.000, None, .922, .885]]
    
    data = mnist_acc if 'mnist' == args.dataset else cifar10_acc
    data = pd.DataFrame(data, columns=('tol',) + models)
    data = data.melt(id_vars='tol', value_vars=models, var_name='model', value_name='accuracy')
    data = data.dropna()
    
    results = results.merge(data, on=('tol', 'model'))
    results = results.replace({'tol': {
            0.0001: r'$10^{-4}$',
            0.0010: r'$10^{-3}$',
            0.0100: r'$10^{-2}$',
            0.1000: r'$10^{-1}$',
            1.0000: r'$10^{0}$',
            10.0000: r'$10^{1}$',
        }})
    
    # print(results)
    
    g = sns.FacetGrid(results, row='epsilon', col='p', hue='model', sharex=False, sharey=False)
    
    # g.map_dataframe(plt.plot, 'successrate', 'accuracy')
    g.map_dataframe(sns.lineplot, 'successrate', 'accuracy', size='tol')
    g.add_legend()
    g.savefig('prova.pdf')



def diff(args):

    dataset = 'MNIST' if 'mnist' in args.run else 'CIFAR-10'
    exps = Experiment.gather(args.run)
    exps = Experiment.filter(args.filter, exps)
    # exps = Experiment.filter({'distance': float('inf')}, exps)
    exps = list(exps)
    
    def _make_plot(d):
        results = Experiment.collect_all(exps, f'diff_{d}.csv')
        
        # XXX TO FIX IN diff.py
        corrupted = results.loc[:, '0.0':'1.0'].isna().all(axis=1)
        if corrupted.any():
            results.loc[corrupted, '0.0':'1.0'] = results.loc[corrupted, '0.0.1':'1.0.1'].values
            results = results.dropna(axis=1)
        
        results = results.sort_values('tol')
        id_cols = [c for c in results.columns if not c.startswith('0') and not c.startswith('1')]
        results = results.melt(id_vars=id_cols, var_name='t', value_name='diff')
        results['t'] = results.t.astype(np.float32)
        results[r'$\tau$'] = results.tol.astype(np.float32).apply(lambda x: rf'$10^{{{np.log10(x).round()}}}$')# .apply(lambda x: rf'$\mathrm{{{x}}}$')
        if d == 'cos':
            results['diff'] = 1 - results['diff']
        plt.figure(figsize=figsize(1))
        ax = sns.lineplot(x='t', y='diff', hue=r'$\tau$', ci=None, data=results) # ci='sd'
        plt.ylabel(r'$|\mathbf{h}(t) - \mathbf{h}_\text{adv}(t)|_2$')
        plt.xlabel(r'$t$')
        plt.xlim(0, 1)
        sns.despine()
        # plt.legend(fontsize='xx-small', title_fontsize='16')
        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_tick_params(direction='out', color=ax.spines['left'].get_ec())
        
        plt.savefig(f'{args.output}_{dataset}.pdf', bbox_inches="tight")
        plt.close()

    _make_plot('l2')
    # _make_plot('cos')


def success_rate(args):
    models = ('resnet', 'mixed', 'fullode')

    def _get_results(model):
        exp_dir = os.path.join(args.root, f'{args.dataset}_{model}', 'adv-attack')
        exps = Experiment.gather(exp_dir)
        exps = Experiment.filter(args.filter, exps)
        results = Experiment.collect_all(exps, 'results.csv')
        results['model'] = model
        return results

    results = map(_get_results, models)
    results = pd.concat(results, axis=0)

    results = results.rename({'distance_x': 'p', 'distance_y': 'distance'}, axis=1)
    results['success_rate'] = np.isfinite(results.distance.values)

    natural_errors = results.distance == 0
    results = results[~natural_errors]  # discard natural errors
    
    good_eps = (0.05, 0.1, 0.3) if args.dataset == 'mnist' else (0.01, 0.03, 0.05)
    results = results[results.epsilon.isin(good_eps)]

    results_l2 = results[results.p == 2.0]
    results_linf = results[results.p == float('inf')]

    def _rate_pivot_table(results, l):
        rate = pd.pivot_table(results, values='success_rate', index='tol', columns=['epsilon', 'model'])

        rate = rate.round(2)
        # rate = rate.rename(columns={'epsilon': r'$\varepsilon$'})
        rate = rate.rename(index={
            'tol': r'$\tau$',
            0.0001: r'$10^{-4}$',
            0.0010: r'$10^{-3}$',
            0.0100: r'$10^{-2}$',
            0.1000: r'$10^{-1}$',
            1.0000: r'$10^{0}$',
            10.0000: r'$10^{1}$',
        }, level=0)
        # rate = rate.rename(columns={2.0: r'$L_2$', float('inf'): r'$L_\infty$'}, level=0)
        rate = rate.rename(columns=lambda x: rf'$\varepsilon = {x}$, $d = {l}$', level=0)
        rate = rate.reindex(models, axis=1, level=1)
        rate = rate.rename(columns={'resnet': r'\multicolumn{1}{r}{Res}', r'mixed': '\multicolumn{1}{r}{Mix}', 'fullode': r'\multicolumn{1}{r}{OON}'}, level=1)
        return rate

    with open(args.output, 'w') as out:
        rate = _rate_pivot_table(results_l2, 'L_2')
        rate.to_latex(out, escape=False, multicolumn_format='c')
        print(rate)

        rate = _rate_pivot_table(results_linf, 'L_\infty')
        rate.to_latex(out, escape=False, multicolumn_format='c')
        print(rate)


def classification_performance(args):
    models = ('resnet', 'mixed', 'fullode')

    exps = Experiment.gather(args.root, main='model')
    exps = Experiment.filter({'dataset': args.dataset}, exps)
    results = Experiment.collect_all(exps, 'nfe.csv.gz')
    results = results[results.t1 == 1]  # keep only complete dynamics
    results['accuracy'] = results.y_pred == results.y_true

    results = results.rename({'tol_x': 'training_tol', 'tol_y': 'tol'}, axis=1)
    rate = pd.pivot_table(results, values='accuracy', index='tol', columns=['model', 'downsample'])
    
    rate = (100 * (1 - rate)).round(2)
    rate = rate.rename(mapper=lambda x: r'$10^{{{}}}$'.format(int(round(np.log10(x)))), level=0)

    with open(args.output, 'w') as out:
        rate.to_latex(out, escape=False, multicolumn_format='c')
        print(rate)
        

def success_rate_single(args):
    exps = Experiment.gather(args.run)
    exps = Experiment.filter(args.filter, exps)
    results = Experiment.collect_all(exps, 'results.csv')
    results = results.rename({'distance_x': 'p', 'distance_y': 'distance'}, axis=1)
    results['success_rate'] = np.isfinite(results.distance.values)

    natural_errors = results.distance == 0
    results = results[~natural_errors]  # discard natural errors

    rate = pd.pivot_table(results, values='success_rate', index='tol', columns=['epsilon', 'p'])

    rate = (rate * 100).round(2)
    rate = rate.rename_axis(columns={'p': r'$p$', 'epsilon': r'$\varepsilon$'})
    rate = rate.rename(columns={2.0: r'$L_2$', float('inf'): r'$L_\infty$'}, level=1)

    print(rate)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Plot adversarial stuff')
    parser.add_argument('-f', '--filter', default={}, type=exp_filter)
    subparsers = parser.add_subparsers()

    parser_sr = subparsers.add_parser('success_rate')
    parser_sr.add_argument('root', type=str)
    parser_sr.add_argument('dataset', type=str)
    parser_sr.add_argument('-o', '--output', default='success_rate.tex')
    parser_sr.set_defaults(func=success_rate)
    
    parser_ar = subparsers.add_parser('accuracy_vs_robustness')
    parser_ar.add_argument('root', type=str)
    parser_ar.add_argument('dataset', type=str)
    parser_ar.add_argument('-o', '--output', default='accuracy_vs_robustness.pdf')
    parser_ar.set_defaults(func=accuracy_vs_robustness)
    
    parser_cp = subparsers.add_parser('classification_performance')
    parser_cp.add_argument('root', type=str)
    parser_cp.add_argument('dataset', type=str)
    parser_cp.add_argument('-o', '--output', default='classification_performance.tex')
    parser_cp.set_defaults(func=classification_performance)

    parser_srs = subparsers.add_parser('success_rate_single')
    parser_srs.add_argument('run', type=str, default='runs/')
    parser_srs.add_argument('-o', '--output', default='success_rate.tex')
    parser_srs.set_defaults(func=success_rate_single)

    parser_diff = subparsers.add_parser('diff')
    parser_diff.add_argument('run', type=str, default='runs/')
    parser_diff.add_argument('-o', '--output', default='diff')
    parser_diff.set_defaults(func=diff)

    args = parser.parse_args()
    args.func(args)
