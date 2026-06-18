import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def _normalize_image(x, device):
    x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
    return x * 2.0 - 1.0


def _to_uint8_image(x):
    x = (x + 1) / 2
    x = x.detach().cpu().numpy().transpose([1, 2, 0])
    x = np.round(np.clip(x * 255, 0, 255)).astype(np.uint8)
    return x[:, :, ::-1]


def _load_ema_params(model_without_ddp):
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)
    return model_state_dict


def _barrier_if_needed():
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if getattr(args, "task", "generation") == "restoration":
            rainy, clean, labels = batch[0], batch[1], batch[2]
            rainy = _normalize_image(rainy, device)
            clean = _normalize_image(clean, device)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model(clean, labels, condition=rainy)
        else:
            x, labels = batch
            # normalize image to [-1, 1]
            x = _normalize_image(x, device)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = _load_ema_params(model_without_ddp)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        _barrier_if_needed()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    _barrier_if_needed()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    _barrier_if_needed()


@torch.no_grad()
def evaluate_restoration(model_without_ddp, data_loader, device, args, epoch, log_writer=None, has_targets=True):
    model_without_ddp.eval()
    local_rank = misc.get_rank()
    save_folder = os.path.join(
        args.output_dir,
        "restoration-{}-steps{}-res{}".format(model_without_ddp.method, model_without_ddp.steps, args.img_size)
    )
    if misc.get_rank() == 0 and args.save_eval_images:
        os.makedirs(save_folder, exist_ok=True)
    _barrier_if_needed()

    model_state_dict = _load_ema_params(model_without_ddp)
    psnr_total = torch.tensor(0.0, device=device)
    count_total = torch.tensor(0.0, device=device)

    for batch_idx, batch in enumerate(data_loader):
        rainy, clean, labels = batch[0], batch[1], batch[2]
        names = batch[3] if len(batch) > 3 else None
        rainy = _normalize_image(rainy, device)
        clean = _normalize_image(clean, device)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            restored = model_without_ddp.restore(rainy, labels)

        if has_targets:
            restored_01 = ((restored.float() + 1) * 0.5).clamp(0, 1)
            clean_01 = ((clean.float() + 1) * 0.5).clamp(0, 1)
            mse = (restored_01 - clean_01).pow(2).flatten(1).mean(dim=1).clamp_min(1e-10)
            psnr = -10.0 * torch.log10(mse)
            psnr_total += psnr.sum()
            count_total += psnr.numel()

        if args.save_eval_images:
            restored_cpu = restored.detach().cpu()
            for b_id in range(restored_cpu.size(0)):
                if names is None:
                    filename = "{}_{}.png".format(str(batch_idx).zfill(5), str(b_id).zfill(3))
                else:
                    filename = names[b_id].replace("\\", "__").replace("/", "__")
                    filename = os.path.splitext(filename)[0] + ".png"
                filename = "{}_rank{}_{}".format(str(batch_idx).zfill(5), local_rank, filename)
                cv2.imwrite(os.path.join(save_folder, filename), _to_uint8_image(restored_cpu[b_id]))

    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(psnr_total)
        torch.distributed.all_reduce(count_total)

    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    _barrier_if_needed()

    if has_targets and count_total.item() > 0:
        psnr_avg = (psnr_total / count_total).item()
        print("Restoration PSNR: {:.4f}".format(psnr_avg))
        if log_writer is not None:
            log_writer.add_scalar('restoration_psnr', psnr_avg, epoch)
