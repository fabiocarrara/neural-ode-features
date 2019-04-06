import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms

from model import ODENet, ResNet


class TinyImageNet200(Dataset):
    def __init__(self, root, download=True, transform=None, target_transform=None, split='train'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_data():
            assert download, "Data is missing or corrupted. Please provide download=True to download."
            print('Downloading TinyImageNet-200 dataset ...')
            raise NotImplementedError("TinyImageNet download not implmented yet, manually download from: "
                                      "http://cs231n.stanford.edu/tiny-imagenet-200.zip")

        assert split in ('train', 'val', 'test'), "Parameter 'split' must be one of: train, val, test"
        self.split = split
        root = os.path.join(root, self.split)

        if self.split == 'train':
            self.dataset = ImageFolder(root, transform=transform, target_transform=target_transform)
        elif self.split == 'val':
            self.image_root = os.path.join(root, 'images')
            annot_file = os.path.join(root, 'val_annotations.txt')
            self.annot = pd.read_csv(annot_file, sep='\t', header=None, usecols=(0, 1), names=('url', 'label'))
            label2idx = {label: i for i, label in enumerate(sorted(self.annot.label.unique().tolist()))}
            self.annot['target'] = self.annot.label.apply(lambda x: label2idx[x])
        else:
            raise NotImplementedError("Loader for the 'test' split has not been implemented yet.")

    def _check_data(self):
        if not os.path.exists(self.root):
            return False

        for split in ('train', 'val', 'test'):
            split_root = os.path.join(self.root, split)
            if not os.path.exists(split_root):
                return False

        return True

    def __getitem__(self, index):
        if self.split == 'train':
            return self.dataset[index]
        elif self.split == 'val':
            filename, target = self.annot.loc[index, ['url', 'target']]
            path = os.path.join(self.image_root, filename)
            image = default_loader(path)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                target = self.target_transform(target)

            return image, target

    def __len__(self):
        if self.split == 'train':
            return len(self.dataset)
        elif self.split == 'val':
            return len(self.annot)


def load_dataset(args):
    if args.dataset == 'mnist':
        if args.augmentation == 'none':
            train_transform = transforms.ToTensor()
        elif args.augmentation == 'crop':
            train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
            ])

        test_transform = transforms.ToTensor()

        train_data = datasets.MNIST('data/mnist', download=True, train=True, transform=train_transform)
        test_data = datasets.MNIST('data/mnist', download=True, train=False, transform=test_transform)
        in_ch = 1
        out = 10

    elif args.dataset == 'cifar10':
        if args.augmentation == 'none':
            train_transform = test_transform = transforms.ToTensor()
        elif args.augmentation == 'crop+flip+norm':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif args.augmentation == 'crop+jitter+flip+norm':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        train_data = datasets.CIFAR10('data/cifar10', download=True, train=True, transform=train_transform)
        test_data = datasets.CIFAR10('data/cifar10', download=True, train=False, transform=test_transform)
        in_ch = 3
        out = 10

    elif args.dataset == 'tiny-imagenet-200':
        if args.augmentation == 'none':
            train_transform = test_transform = transforms.ToTensor()
        elif args.augmentation == 'crop+flip+norm':
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])

        elif args.augmentation == 'crop+jitter+flip+norm':
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])

        train_data = TinyImageNet200('data/tiny-imagenet-200', download=True, split='train', transform=train_transform)
        test_data = TinyImageNet200('data/tiny-imagenet-200', download=True, split='val', transform=test_transform)
        in_ch = 3
        out = 200

    return train_data, test_data, in_ch, out


def load_test_data(exp):
    params = next(exp.params.itertuples())

    if params.dataset == 'mnist':
        test_data = datasets.MNIST('data/mnist', download=True, train=False, transform=transforms.ToTensor())

    elif params.dataset == 'cifar10':
        if params.augmentation == 'none':
            test_transform = transforms.ToTensor()

        elif params.augmentation in ('crop+flip+norm', 'crop+jitter+flip+norm'):
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        test_data = datasets.CIFAR10('data/cifar10', download=True, train=False, transform=test_transform)

    elif params.dataset == 'tiny-imagenet-200':
        if params.augmentation == 'none':
            test_transform = transforms.ToTensor()

        elif params.augmentation in ('crop+flip+norm', 'crop+jitter+flip+norm'):
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])
        test_data = TinyImageNet200('data/tiny-imagenet-200', download=True, split='val', transform=test_transform)


    return test_data


def load_model(exp, in_ch=None):
    params = next(exp.params.itertuples())

    if in_ch is None:
        in_ch = 1 if params.dataset == 'mnist' else 3

    out = 200 if params.dataset == 'tiny-imagenet-200' else 10

    if params.model == 'odenet':
        model = ODENet(in_ch, out=out, n_filters=params.filters, downsample=params.downsample, method=params.method, tol=params.tol,
                       adjoint=params.adjoint, dropout=params.dropout)
    else:
        model = ResNet(in_ch, out=out, n_filters=params.filters, downsample=params.downsample, dropout=params.dropout)

    checkpoint = torch.load(exp.ckpt())['model']  # get best model
    model.load_state_dict(checkpoint)

    return model


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    d = TinyImageNet200('data/tiny-imagenet-200', split='train', transform=transforms.ToTensor())
    d = DataLoader(d, batch_size=100, shuffle=False, num_workers=4)

    mu = torch.zeros(3)
    std = torch.zeros(3)

    for x, _ in tqdm(d):
        mu += x.sum(0).sum(-1).sum(-1)

    mu /= len(d.dataset) * (64 * 64)
    print(mu.tolist())

    for x, _ in tqdm(d):
        std += ((x - mu.view(-1,1,1))**2).sum(0).sum(-1).sum(-1)

    std = torch.sqrt(std / (len(d.dataset) * (64 * 64)))
    print(std.tolist())



