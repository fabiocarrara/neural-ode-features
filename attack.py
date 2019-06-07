import argparse
import os
import foolbox
import numpy as np
import pandas as pd
import time

from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

import utils
from expman import Experiment

def main(args):
    exp = Experiment.from_dir(args.run, main='model')
    params = next(exp.params.itertuples())

    # data setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())
    ])
    data = MNIST('data/mnist', download=True, train=False, transform=transform)

    # model setup
    model = utils.load_model(exp).eval().cuda()
    if args.tol is None:
        args.tol = params.tol

    if params.model == "odenet":
        model.odeblock.tol = args.tol

    preproc = utils.PREPROC['mnist']
    fmodel = foolbox.models.PyTorchModel(model, bounds=preproc, num_classes=10, preprocessing=preproc)

    # attack setup
    if args.distance == '2':
        attack = foolbox.attacks.L2BasicIterativeAttack
        distance = foolbox.distances.MSE
    elif args.distance == 'inf':
        attack = foolbox.attacks.LinfinityBasicIterativeAttack
        distance = foolbox.distances.Linf
    
    attack = attack(fmodel, distance=distance)
    
    sub_exp_root = exp.path_to('adv-attack')
    os.makedirs(sub_exp_root, exist_ok=True)

    sub_exp = Experiment(args, root=sub_exp_root, ignore=('run',))
    print(sub_exp)
    results_file = sub_exp.path_to('results.csv')
    results = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    # perform attack
    progress = tqdm(data)
    for i, (image, label) in enumerate(progress):
        if not results.empty and i in results.sample_id.values:
           continue
        
        start = time.time()
        adversarial = attack(image, label, unpack=False, binary_search=False, epsilon=args.epsilon)
        elapsed = time.time() - start
        result = pd.DataFrame(dict(
            sample_id         = i,
            label             = label,
            elapsed_time      = elapsed,
            distance          = adversarial.distance.value,
            adversarial_class = adversarial.adversarial_class,
            original_class    = adversarial.original_class,
        ), index=[0])
        
        results = results.append(result, ignore_index=True)
        results.to_csv(results_file, index=False)
        
        success = (results.adversarial_class != results.original_class)
        successes = success.sum()
        success_rate = success.mean()
        
        progress.set_postfix({'success_rate': f'{success_rate:.2%} ({successes}/{len(success)})'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attacks on Neural ODEs')
    parser.add_argument('run', help='Run to attack')
    parser.add_argument('-t', '--tol', type=float, default=None, help='ODE solver tolerance')
    parser.add_argument('-a', '--attack', choices=('pgd',), default='pgd', help='Attack to perform')
    parser.add_argument('-d', '--distance', choices=('2', 'inf'), default='inf', help='L_p distance')
    parser.add_argument('-e', '--epsilon', type=float, default=0.3, help='epsilon')
    
    
    

    args = parser.parse_args()
    main(args)
