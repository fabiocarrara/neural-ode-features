import argparse
import os
import time

import torch
import torch.nn.functional as F
import foolbox
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST, CIFAR10
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

    preproc = utils.PREPROC[params.dataset]
    if params.dataset == 'mnist':
        data = MNIST('data/mnist', download=True, train=False, transform=transform)
    elif params.dataset == 'cifar10':
        data = CIFAR10('data/cifar10', download=True, train=False, transform=transform)
        preproc = map(lambda x: np.array(x).reshape((3, 1, 1)), preproc)  # expand dimensions
        preproc = tuple(preproc)

    t = np.linspace(0, 1, args.resolution + 1).tolist()

    # model setup
    model = utils.load_model(exp).eval().cuda()
    extractor = utils.load_model(exp).eval().cuda()
    extractor.to_features_extractor(keep_pool=False)
    extractor.odeblock.t1 = t

    if args.tol is None:
        args.tol = params.tol

    if params.model == 'odenet':
        model.odeblock.tol = args.tol
        extractor.odeblock.tol = args.tol

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preproc)

    # attack setup
    if args.distance == 2:
        attack = foolbox.attacks.L2BasicIterativeAttack
        distance = foolbox.distances.MSE
    elif args.distance == float('inf'):
        attack = foolbox.attacks.LinfinityBasicIterativeAttack
        distance = foolbox.distances.Linf

    attack = attack(fmodel, distance=distance)

    sub_exp_root = exp.path_to('adv-attack')
    os.makedirs(sub_exp_root, exist_ok=True)

    sub_exp = Experiment(args, root=sub_exp_root, ignore=('run', 'resolution'))
    print(sub_exp)
    results_file = sub_exp.path_to('results.csv')
    diff_l2_file = sub_exp.path_to('diff_l2.csv')
    diff_cos_file = sub_exp.path_to('diff_cos.csv')

    if not os.path.exists(results_file):
        print('No results on attacks found:', results_file)
        return

    results = pd.read_csv(results_file).set_index('sample_id')

    diff_l2 = pd.read_csv(diff_l2_file) if os.path.exists(diff_l2_file) else pd.DataFrame()
    diff_cos = pd.read_csv(diff_cos_file) if os.path.exists(diff_cos_file) else pd.DataFrame()
    diff_cols = ['sample_id'] + t

    progress = tqdm(data)
    for i, (image, label) in enumerate(progress):
        
        if (not diff_l2.empty and not diff_cos.empty and
            i in diff_l2.sample_id.values and i in diff_cos.sample_id.values):
            continue  # skipping, already computed

        perturbation_distance = results.at[i, 'distance']
        if perturbation_distance == 0 or not np.isfinite(perturbation_distance):
            continue  # skipping natural errors or not-found adversarials

        if not isinstance(label, int):
            label = label.item()

        start = time.time()
        adversarial = attack(image, label, unpack=False, binary_search=False, epsilon=args.epsilon)
        elapsed = time.time() - start

        if adversarial.perturbed is None:
            tqdm.write(f'WARN: adversarial not found when reproducing [sample_id = {i}]')
            continue

        with torch.no_grad():
            original_image = torch.from_numpy(adversarial.unperturbed).cuda()
            original_traj = extractor(original_image.unsqueeze(0))

            adversarial_image = torch.from_numpy(adversarial.perturbed).cuda()
            adversarial_traj = extractor(adversarial_image.unsqueeze(0))

        adversarial_traj = adversarial_traj.reshape(args.resolution + 1, -1)
        original_traj = original_traj.reshape(args.resolution + 1, -1)

        """ L2 """
        diff_traj = adversarial_traj - original_traj
        diff_traj = (diff_traj ** 2).sum(1).sqrt()
        diff_traj = diff_traj.cpu().numpy()
        tmp = pd.DataFrame([[i] + diff_traj.tolist()], columns=diff_cols)
        diff_l2 = diff_l2.append(tmp, ignore_index=True)
        diff_l2.to_csv(diff_l2_file, index=False)
        
        """ Cosine similarity """
        diff_traj = F.cosine_similarity(adversarial_traj, original_traj)
        diff_traj = diff_traj.cpu().numpy()
        tmp = pd.DataFrame([[i] + diff_traj.tolist()], columns=diff_cols)
        diff_cos = diff_cos.append(tmp, ignore_index=True)
        diff_cos.to_csv(diff_cos_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attacks on Neural ODEs')
    parser.add_argument('run', help='Run to attack')
    parser.add_argument('-t', '--tol', type=float, default=None, help='ODE solver tolerance')
    parser.add_argument('-a', '--attack', choices=('pgd',), default='pgd', help='Attack to perform')
    parser.add_argument('-d', '--distance', type=float, choices=(2.0, float('inf')), default=2.0, help='L_p distance')
    parser.add_argument('-e', '--epsilon', type=float, default=0.05, help='epsilon')
    parser.add_argument('-s', '--stepsize', type=float, default=0.05, help='step size for iterative attacks')
    parser.add_argument('-r', '--resolution', type=float, default=50, help='n. sampling points')

    args = parser.parse_args()
    main(args)
