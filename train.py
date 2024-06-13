import argparse
import torch
from trainer_class import Trainer


if __name__ == "__main__":
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # Working directory: experiments/exp_name
    parser.add_argument("--exp_name", type=str, required=True)
    # Training dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    # Select 'lmdb' for the lmdb files, like LSUN (https://github.com/fyu/lsun)
    # Select 'normal' for the dataset storing files (e.g., in PNG format) in a folder, like FFHQ (https://github.com/NVlabs/ffhq-dataset)
    parser.add_argument("--dataset_type", choices=['lmdb', 'normal'], required=True)

    # Exp   Exp_fun   None
    parser.add_argument("--train_exp_mode", type=str, default=None)
    # Wass
    parser.add_argument("--train_wass_mode", type=str, default=None)
    parser.add_argument("--save_model_part", type=bool, default=False)
    # We recommend training at least 80k iterations
    parser.add_argument("--num_iters", type=int, required=True)
    # Hyper-parameters
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--lambda_Ex", type=float, default=10)
    parser.add_argument("--clamp", type=float, default=-1.0)

    # Resume training
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--load_ckpt_dir", type=str, default=None)

    # Trainig parameters
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--real_r1", type=float, default=10)
    parser.add_argument("--texture_r1", type=float, default=1)
    parser.add_argument("--dist_r1", type=float, default=1)
    parser.add_argument("--ref_crop", type=int, default=4)
    parser.add_argument("--n_crop", type=int, default=8)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--structure_channel", type=int, default=8)
    parser.add_argument("--texture_channel", type=int, default=2048)
    
    
    # Output logs every 'log_every' iterations
    parser.add_argument("--log_every", type=int, default=20)
    # Save example images every 'show_every' iterations
    parser.add_argument("--show_every", type=int, default=100)
    # Save models every 'save_every' iterations
    parser.add_argument("--save_every", type=int, default=200000)

    args = parser.parse_args()
    args.start_iter = 0
    args.blur_kernel = (1, 3, 3, 1)

    trainer = Trainer(args)
    trainer.train()


