import argparse
import copy
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from dataset import (
    PairedRainDataset,
    ScenePairedRainDataset,
    SceneValPatchDataset,
    ScenePairedRainDatasetV2,
    SceneValPatchDatasetV2,
    ValPatchDataset,
)
from denoiser import Denoiser
from engine_jit import evaluate_best_metric, train_one_epoch
from loss import DynamicRaindropLoss
from util.crop import center_crop_arr


def set_seed(seed=42):
    # 1. 设置 Python 环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 对于 PyTorch 1.8+，强制 cuBLAS 使用确定性算法
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. 设置 Python 内置随机种子
    random.seed(seed)

    # 3. 设置 Numpy 随机种子
    np.random.seed(seed)

    # 4. 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 5. 配置 cuDNN
    # 禁用 cuDNN 自动寻找最快算法的机制（因为最快算法往往有随机性）
    torch.backends.cudnn.benchmark = False
    # 强制 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True

    # 6. 强制 PyTorch 使用确定性算法 (可选，可能会导致程序报错如果使用了不支持确定性的操作)
    # torch.use_deterministic_algorithms(True)


def get_args_parser():
    parser = argparse.ArgumentParser("JiT", add_help=False)

    # architecture
    parser.add_argument(
        "--model",
        default="JiT-B/16",
        type=str,
        metavar="MODEL",
        help="Name of the model to train",
    )
    parser.add_argument("--img_size", default=256, type=int, help="Image size")
    parser.add_argument(
        "--attn_dropout", type=float, default=0.0, help="Attention dropout rate"
    )
    parser.add_argument(
        "--proj_dropout", type=float, default=0.0, help="Projection dropout rate"
    )
    parser.add_argument("--use_bg_subnet", type=int, default=0, help="use_bg_subnet")
    parser.add_argument(
        "--use_scene_dataset", type=int, default=0, help="use_scene_dataset"
    )
    parser.add_argument(
        "--scene_train_path",
        default="",
        type=str,
        help="Path to scene labels for train set. Supports csv for V1 or json for V2 datasets.",
    )
    parser.add_argument(
        "--scene_val_path",
        default="",
        type=str,
        help="Path to scene labels for validation set. Defaults to scene_train_path.",
    )
    parser.add_argument("--eval_epoch", type=int, default=5, help="eval_epoch")
    # training
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="Epochs to warm up LR"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU (effective batch size = batch_size * # GPUs)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, metavar="LR", help="Learning rate (absolute)"
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-5,
        metavar="LR",
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="Minimum LR for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--lr_schedule", type=str, default="constant", help="Learning rate schedule"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay (default: 0.0)"
    )
    parser.add_argument(
        "--ema_decay1",
        type=float,
        default=0.9999,
        help="The first ema to track. Use the first ema for sampling by default.",
    )
    parser.add_argument(
        "--ema_decay2", type=float, default=0.9996, help="The second ema to track"
    )
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="Starting epoch"
    )
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for faster GPU transfers",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument(
        "--sampling_method", default="heun", type=str, help="ODE samping method"
    )
    parser.add_argument(
        "--num_sampling_steps", default=50, type=int, help="Sampling steps"
    )
    parser.add_argument(
        "--cfg", default=2.9, type=float, help="Classifier-free guidance factor"
    )
    parser.add_argument(
        "--interval_min", default=0.1, type=float, help="CFG interval min"
    )
    parser.add_argument(
        "--interval_max", default=1.0, type=float, help="CFG interval max"
    )
    parser.add_argument(
        "--eval_num_images",
        default=50000,
        type=int,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--eval_freq", type=int, default=40, help="Frequency (in epochs) for evaluation"
    )
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--evaluate_gen", action="store_true")
    parser.add_argument(
        "--gen_bsz", type=int, default=128, help="Generation batch size"
    )

    # dataset
    parser.add_argument(
        "--data_path",
        default=r"/scrinvme/huilin/bdd/cp_data/raindrop_remove_2026/RainDrop_Train2",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--val_data_path",
        default="",
        type=str,
        help="Validation dataset path. Defaults to data_path.",
    )
    parser.add_argument("--class_num", default=1000, type=int)

    # checkpointing
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Directory to save outputs (empty for no saving)",
    )
    parser.add_argument(
        "--resume", default="", help="Folder that contains checkpoint to resume from"
    )
    parser.add_argument(
        "--save_last_freq",
        type=int,
        default=5,
        help="Frequency (in epochs) to save checkpoints",
    )
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training/testing"
    )

    # distributed training
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print("Job directory:", os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )

    if not args.use_scene_dataset:
        print("[DATASET] use PairedRainDataset")
        dataset_train = PairedRainDataset(
            rain_dir=os.path.join(args.data_path, "Drop"),
            clean_dir=os.path.join(args.data_path, "Clear"),
            transform=transform_train,
        )
    else:
        print("[DATASET] use ScenePairedRainDatasetV2")
        dataset_train = ScenePairedRainDatasetV2(
            rain_dir=os.path.join(args.data_path, "Drop"),
            clean_dir=os.path.join(args.data_path, "Clear"),
            transform=transform_train,
            scene_path=args.scene_train_path or None,
        )

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_data_path = args.val_data_path or args.data_path
    if not args.use_scene_dataset:
        print("[DATASET] use ValPatchDataset")
        dataset_val_full = ValPatchDataset(
            rain_dir=os.path.join(val_data_path, "Drop"),
            clean_dir=os.path.join(val_data_path, "Clear"),
        )
    else:
        print("[DATASET] use SceneValPatchDatasetV2")
        dataset_val_full = SceneValPatchDatasetV2(
            rain_dir=os.path.join(val_data_path, "Drop"),
            clean_dir=os.path.join(val_data_path, "Clear"),
            scene_path=args.scene_val_path or args.scene_train_path or None,
        )

    num_val_images = min(args.eval_num_images, len(dataset_val_full))
    step = len(dataset_val_full) // num_val_images
    indices = [i * step for i in range(num_val_images)]
    dataset_val = torch.utils.data.Subset(dataset_val_full, indices)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=max(args.batch_size // 4, 2),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    model = Denoiser(args)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {model}, number of trainable parameters: {n_params / 1e6:.6f}M")

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if getattr(args, "distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    # )
    print(optimizer)
    criterion = DynamicRaindropLoss(
        device=device,
        target_w_rec=1.0,
        target_w_ssim=1.0,  # 对应指标公式的 10
        target_w_lpips=0.5,  # 对应指标公式的 5
        total_epochs=args.epochs,
    )

    checkpoint_path = (
        os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        ema_state_dict1 = checkpoint["model_ema1"]
        ema_state_dict2 = checkpoint["model_ema2"]
        model_without_ddp.ema_params1 = [
            ema_state_dict1[name].to(device)
            if "bg_subnet" not in name
            else model_without_ddp.state_dict()[name]
            for name, _ in model_without_ddp.named_parameters()
        ]
        model_without_ddp.ema_params2 = [
            ema_state_dict2[name].to(device)
            if "bg_subnet" not in name
            else model_without_ddp.state_dict()[name]
            for name, _ in model_without_ddp.named_parameters()
        ]
        print("Resumed checkpoint from", args.resume)

        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(
            list(model_without_ddp.parameters())
        )
        model_without_ddp.ema_params2 = copy.deepcopy(
            list(model_without_ddp.parameters())
        )
        print("Training from scratch")

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    best_score = -float("inf")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            criterion,
            model_without_ddp,
            data_loader_train,
            optimizer,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
        )

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
            )

        if epoch % 50 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
            )
        if args.online_eval and (epoch % args.eval_epoch == 0 or epoch + 1 == args.epochs):
            total_scores = evaluate_best_metric(
                model_without_ddp, data_loader_val, device, steps=args.num_sampling_steps
            )
            print(
                f"Epoch [{epoch}] - "
                f"score: {total_scores['score']:.4f}, "
                f"psnr_y: {total_scores['psnr']:.4f}, "
                f"ssim_y: {total_scores['ssim']:.4f}, "
                f"lpips: {total_scores['lpips']:.4f}"
            )

            if log_writer is not None:
                log_writer.add_scalar(
                    "val_composite_score", total_scores["score"], epoch
                )
                log_writer.add_scalar("val_psnr_y", total_scores["psnr"], epoch)
                log_writer.add_scalar("val_ssim_y", total_scores["ssim"], epoch)
                log_writer.add_scalar("val_lpips", total_scores["lpips"], epoch)

            if total_scores["score"] > best_score:
                best_score = total_scores["score"]
                print(f"New Best score: {best_score:.4f}, saving...")
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    epoch_name="best",
                )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

        # if epoch >= args.warmup_epochs:
        #     scheduler.step()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time:", total_time_str)


if __name__ == "__main__":
    set_seed(42)
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
