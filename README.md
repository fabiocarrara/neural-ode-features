# Neural ODE Image Classifiers

Pytorch code for training and evaluating Neural ODEs image classifiers on MNIST and CIFAR-10 datasets.
It reproduces experiments presented in the following papers:

> \[1\] Carrara, F., Amato, G., Falchi, F. and Gennaro, C., 2019, September. Evaluation of Continuous Image Features Learned by ODE Nets. In *International Conference on Image Analysis and Processing (ICIAP '19)* (pp. 432-442). Springer, Cham.
> 
> \[2\] Carrara, F., Amato, G., Falchi, F. and Gennaro, C., 2020, June. Continuous ODE-defined Image Features for Adaptive Retrieval. In *Proceedings of the 2020 International Conference on Multimedia Retrieval (ICMR '20)* (pp. 198-206). ACM.
>
> \[3\] Carrara, F., Caldelli, R., Falchi, F. and Amato, G., 2019, December. On the robustness to adversarial examples of neural ode image classifiers. In *2019 IEEE International Workshop on Information Forensics and Security (WIFS '19)* (pp. 1-6). IEEE.

## Getting Started

Clone and install requirements:

```bash
git clone --recursive https://github.com/fabiocarrara/neural-ode-features.git
cd neural-ode-features
pip install -e torchdiffeq
pip install torchvision foolbox h5py pandas tqdm seaborn sklearn
```

## Reproduce Experiments

To obtain the trained models and reproduce the experiments described in \[1\] and \[2\], run

```bash
./reproduce.sh
```

Pre-trained models are also available: [neural-ode-features-runs.zip (172MB)](https://drive.google.com/open?id=1nFsG48Kqk-KQzYSyQNRWQg7D9gZMhrjh)

---

To reproduce experiments described in \[3\], obtain the trained models, and then run

```bash
cd adversarial
./reproduce.sh <path/to/specific_run_folder>
```

to attack a specific model and collect results.
