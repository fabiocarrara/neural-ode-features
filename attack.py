import argparse

import foolbox
import numpy as np

import utils
from expman import Experiment


def main(args):
    exp = Experiment.from_dir(args.run, main='model')
    model = utils.load_model(exp).eval()

    preproc = utils.PREPROC['mnist']
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preproc)

    image, label = foolbox.utils.samples('mnist', index=7, data_format='channel_first')
    image /= 255

    pred = fmodel.forward_one(image)
    print(pred, np.argmax(pred), label)

    attack = foolbox.attacks.PGD(fmodel)
    adversarial = attack(image, label)
    print(adversarial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attacks on Neural ODEs')
    parser.add_argument('run', type=str, help='Run to attack')

    args = parser.parse_args()
    main(args)
