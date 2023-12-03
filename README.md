## Requirements

Version number after each item is only the development environment configuration, not the *minimum*, *maximum* or *exact* version you need to run.

The name presented in code format is an *Anaconda* (*Miniconda*) or *PyPi* package name.

  - Python `python` <sup>3.11.6</sup>
  - PyTorch `pytorch` <sup>2.1.0</sup>
  - `pytorch-cuda` <sup>12.1</sup> or `cpuonly` <sup>2.0</sup>
  - PyTorch Geometric `pyg` <sup>2.4.0</sup>
  - TorchMetrics `torchmetrics` <sup>1.2.1</sup>
  - TensorBoard `tensorboard` <sup>2.15.1</sup>
  - `torch-tb-profiler` <sup>0.4.3</sup>
  - `pandas` <sup>2.1.3</sup>
  - `tqdm` <sup>4.66.1</sup>
  - `bitarray` <sup>2.8.3</sup>

Using `pytorch-cuda` is recommended for nVIDIA GPU. `cpuonly` is just for test when you don't have an nVIDIA GPU.

**PyTorch Geometric** does not officially support ROCm in a stable way yet, so a way to install it on AMD GPU would not be given. You may have to figure it out by yourself if you need it.

More information please check [*PyTorch: Get Started*](https://pytorch.org/get-started/locally/) and [*PyTorch Geometric: Installation*](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

You can use *Anaconda* (*Miniconda*) to install requirements when you have an nVIDIA GPU:

```
conda env create -f requirements-cuda.yml
```

or you just want to test on CPU:

```
conda env create -f requirements-cpuonly.yml
```

You can change environment name by modifying the string after `name:` at the 1st line of `requirements-*.yml`.