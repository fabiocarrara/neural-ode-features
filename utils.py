import torch
from torchvision import datasets
from torchvision.transforms import transforms

from model import ODENet, ResNet


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

    return train_data, test_data, in_ch


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

    return test_data


def load_model(exp, in_ch=None):
    params = next(exp.params.itertuples())

    if in_ch is None:
        in_ch = 1 if params.dataset == 'mnist' else 3

    if params.model == 'odenet':
        model = ODENet(in_ch, n_filters=params.filters, downsample=params.downsample, tol=params.tol,
                       adjoint=params.adjoint)
    else:
        model = ResNet(in_ch, n_filters=params.filters, downsample=params.downsample)

    checkpoint = torch.load(exp.ckpt())['model']  # get best model
    model.load_state_dict(checkpoint)

    return model
