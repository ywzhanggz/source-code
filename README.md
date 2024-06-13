## Requirements
* Only Linux is supported. 
* Ninja >= 1.10.2, GCC/G++ >= 9.4.0.
* One high-end NVIDIA GPU with at least 11GB of memory. We have done all development and testing using a A10.
* Python >= 3.7 and PyTorch >= 1.6.0. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 10.2 or later.
* Python libraries: `pip install lmdb imutils opencv-python pandas tqdm`. We use the Anaconda3 2020.11 distribution which installs most of these by default.

## Training
Train a model using the dataset with path of `PATH` and type of `TYPE`.   
The training configuration can be customized with command line option:

| args             | Description                                                                                          |
|:-----------------|:-----------------------------------------------------------------------------------------------------|
| `exp_name`       | The working directory `./experiments/{exp_name}`.                                                    |
| `dataset_type`   | The type of dataset. Select `lmdb` for LMDB files, or `normal` for the folder storing files.         |
| `dataset_path`   | The path of dataset.                                                                                 |
| `num_iters`      | Num of training iterations.                                                                          |
| `N`, `lambda_Ex` | The hyper-parameters of IDEAS.                                                                       |
| `ckpt`           | Train from scratch if ignored, else resume training from `./experiments/NAME/checkpoints/{ckpt}.pt`. |
| `log_every`      | Output logs every `log_every` iterations.                                                            |
| `show_every`     | Save example images every `show_every` iterations under `./experiments/NAME/samples/`.               |
| `save_every`     | Save models every `save_every` iterations under `./experiments/NAME/checkpoints/`.                   |
| `train_exp_mode` | Set the carrier function through the `train_exp_mode` parameter.               					  |
| `train_wass_mode`| Determine whether to use Wasserstein distance through `train_wass_mode` parameter.                   |
