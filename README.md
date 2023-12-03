## Requirements

  - python
  - pytorch
  - pyg
  - torchmetrics
  - tqdm
  - tensorboard
  - pandas
  - bitarray
  - pytorch-cuda *or* cpuonly
  - torch-tb-profiler

Using `pytorch-cuda` is recommended for nVIDIA GPU. `cpuonly` is just for test when you don't have an nVIDIA GPU. For AMD GPU installation and more information please check [*PyTorch: Get Started*](https://pytorch.org/get-started/locally/).

You can use `conda` to install requirements when you have an nVIDIA GPU:

```
conda env create -f requirements-cuda.yml
```

or you just want to test on CPU:

```
conda env create -f requirements-cpuonly.yml
```
