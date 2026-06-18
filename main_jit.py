import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.paired_dataset import PairedImageDataset

import copy
from engine_jit import train_one_epoch, evaluate, evaluate_restoration

from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--task', default='generation', choices=['generation', 'restoration'],
                        help='Train class-conditional generation or paired image restoration')
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--restoration_label_drop_prob', default=0.0, type=float,
                        help='Label dropout used only for restoration task')

    # restoration
    parser.add_argument('--rainy_dir', default='', type=str,
                        help='Training degraded/rainy image directory. Defaults to data_path/Drop or data_path/train/Drop.')
    parser.add_argument('--clean_dir', default='', type=str,
                        help='Training clean image directory. Defaults to data_path/Clear or data_path/train/Clear.')
    parser.add_argument('--val_rainy_dir', default='', type=str,
                        help='Validation/test rainy image directory for restoration evaluation.')
    parser.add_argument('--val_clean_dir', default='', type=str,
                        help='Validation clean image directory. Leave empty to only save restored images.')
    parser.add_argument('--resize_size', default=0, type=int,
                        help='Resize paired restoration images to this square size before crop; 0 keeps aspect ratio.')
    parser.add_argument('--no_hflip', action='store_false', dest='hflip')
    parser.add_argument('--no_vflip', action='store_false', dest='vflip')
    parser.add_argument('--no_rot90', action='store_false', dest='rot90')
    parser.set_defaults(hflip=True, vflip=True, rot90=True)
    parser.add_argument('--restoration_bridge', default='condition', choices=['condition', 'noise'],
                        help='condition starts sampling from the rainy image; noise keeps the original diffusion source.')
    parser.add_argument('--condition_noise_scale', default=0.0, type=float,
                        help='Optional noise added to the rainy source during condition-bridge training.')
    parser.add_argument('--predict_residual', action='store_true', default=True,
                        help='Predict residual and add it to the rainy image for restoration.')
    parser.add_argument('--no_predict_residual', action='store_false', dest='predict_residual')
    parser.add_argument('--recon_l1_weight', default=1.0, type=float)
    parser.add_argument('--residual_l1_weight', default=0.2, type=float)
    parser.add_argument('--ssim_loss_weight', default=0.1, type=float)
    parser.add_argument('--gradient_loss_weight', default=0.05, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--evaluate_restoration', action='store_true')
    parser.add_argument('--save_eval_images', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def _first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return str(path)
    return ''


def _resolve_restoration_dirs(args, split):
    root = Path(args.data_path)
    if split == 'train':
        rainy_dir = args.rainy_dir or _first_existing([
            root / 'train' / 'Drop',
            root / 'Drop',
            root / 'train' / 'rainy',
            root / 'rainy',
            root / 'train' / 'input',
            root / 'input',
        ])
        clean_dir = args.clean_dir or _first_existing([
            root / 'train' / 'Clear',
            root / 'Clear',
            root / 'train' / 'clean',
            root / 'clean',
            root / 'train' / 'gt',
            root / 'gt',
            root / 'train' / 'target',
            root / 'target',
        ])
        if not rainy_dir or not clean_dir:
            raise RuntimeError(
                'Restoration training needs paired directories. Set --rainy_dir and --clean_dir, '
                'or arrange data as data_path/Drop and data_path/Clear.'
            )
        return rainy_dir, clean_dir

    rainy_dir = args.val_rainy_dir or _first_existing([
        root / 'val' / 'Drop',
        root / 'validation' / 'Drop',
        root / 'test' / 'Drop',
        root / 'val' / 'rainy',
        root / 'validation' / 'rainy',
        root / 'test' / 'rainy',
    ])
    clean_dir = args.val_clean_dir or _first_existing([
        root / 'val' / 'Clear',
        root / 'validation' / 'Clear',
        root / 'val' / 'clean',
        root / 'validation' / 'clean',
        root / 'val' / 'gt',
        root / 'validation' / 'gt',
    ])
    return rainy_dir, clean_dir


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.task == 'restoration':
        args.label_drop_prob = args.restoration_label_drop_prob
        if args.class_num == 1000:
            args.class_num = 1
            print("Restoration task detected: using class_num=1 by default.")

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    dataset_val = None
    has_val_targets = False
    if args.task == 'restoration':
        train_rainy_dir, train_clean_dir = _resolve_restoration_dirs(args, 'train')
        dataset_train = PairedImageDataset(
            train_rainy_dir,
            train_clean_dir,
            img_size=args.img_size,
            train=True,
            resize_size=args.resize_size,
            hflip=args.hflip,
            vflip=args.vflip,
            rot90=args.rot90,
        )
        val_rainy_dir, val_clean_dir = _resolve_restoration_dirs(args, 'val')
        if val_rainy_dir:
            dataset_val = PairedImageDataset(
                val_rainy_dir,
                val_clean_dir if val_clean_dir else None,
                img_size=args.img_size,
                train=False,
                resize_size=args.resize_size,
                hflip=False,
                vflip=False,
                rot90=False,
            )
            has_val_targets = bool(val_clean_dir)
        print("Restoration train set:", dataset_train)
        if dataset_val is not None:
            print("Restoration eval set:", dataset_val)
    else:
        # Data augmentation transforms
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ])

        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_val = None
    if dataset_val is not None:
        if args.distributed:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation/restoration
    if args.evaluate_gen:
        if args.task != 'generation':
            raise RuntimeError("--evaluate_gen is only valid for generation. Use --evaluate_restoration for restoration.")
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    if args.evaluate_restoration:
        if args.task != 'restoration':
            raise RuntimeError("--evaluate_restoration requires --task restoration.")
        if data_loader_val is None:
            raise RuntimeError("Restoration evaluation needs --val_rainy_dir or a val/test rainy directory under --data_path.")
        print("Evaluating restoration checkpoint at {} epoch".format(args.start_epoch))
        with torch.no_grad():
            evaluate_restoration(
                model_without_ddp, data_loader_val, device, args, 0,
                log_writer=log_writer, has_targets=has_val_targets
            )
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                if args.task == 'restoration':
                    if data_loader_val is not None:
                        evaluate_restoration(
                            model_without_ddp, data_loader_val, device, args, epoch,
                            log_writer=log_writer, has_targets=has_val_targets
                        )
                else:
                    evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
