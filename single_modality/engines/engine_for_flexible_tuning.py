import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import random
import torch.nn.functional as F
import torch.distributed as dist


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(
    model: torch.nn.Module, criterion: torch.nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
    num_training_steps_per_epoch=None, update_freq=None, args=None,
    bf16=False,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:

            B, C, T, H, W = samples.shape
    
            samples = samples.to(device, non_blocking=True)

            nums_token = T * (H // 14) ** 2
            samples = samples.bfloat16() if bf16 else samples.half()
            
            def compute_mask_ratio(num_current_token, num_input_token):
                if num_current_token <= num_input_token:
                    mask_ratio = 0.0
                else:
                    mask_ratio = 1 - num_input_token / num_current_token
                return mask_ratio

            def compute_smooth_l1_loss(src, tgt):
                tgt = tgt.detach()
                src = src / src.norm(dim=-1, keepdim=True).to(torch.float32)
                tgt = tgt / tgt.norm(dim=-1, keepdim=True).to(torch.float32)
                return torch.nn.SmoothL1Loss()(src, tgt.detach())

            mask_ratio = compute_mask_ratio(nums_token, args.largest_num_input_token)
            outputs1, x_final_1 = model(samples, masking_ratio=mask_ratio, output_head=0, return_cls=True)

            mask_ratio = compute_mask_ratio(nums_token, args.middle_num_input_token)
            outputs2, x_final_2 = model(samples, masking_ratio=mask_ratio, output_head=1, return_cls=True, return_projected=True, align_proj=0)

            mask_ratio = compute_mask_ratio(nums_token, args.least_num_input_token)
            outputs3, x_final_3 = model(samples, masking_ratio=mask_ratio, output_head=2, return_cls=True, return_projected=True, align_proj=1)

            pred = criterion(outputs1, targets) + criterion(outputs2, targets) + criterion(outputs3, targets)
            align_loss = compute_smooth_l1_loss(x_final_2, x_final_1) + compute_smooth_l1_loss(x_final_3, x_final_1)

            loss = pred + align_loss

        else:
            raise NotImplementedError

        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)
        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()

        if loss_list_isnan or loss_list_isinf:
            print("Loss is {}, stopping training".format(loss_value), force=True)
            model.zero_grad()
            continue

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, args, ds=False, bf16=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    current_tokens = args.eval_true_frame * ((args.eval_input_size // 14) ** 2)
    
    if current_tokens <= args.eval_max_token_num:
        mask_ratio = None
        nums_token = current_tokens
    else:
        mask_ratio = 1 - args.eval_max_token_num / current_tokens
        nums_token = args.eval_max_token_num

    print('Nums Token: ', nums_token)

    num_merging_to = args.eval_num_merging_to

    print("Num Merging Ratio: ", num_merging_to)

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos, num_merging_to=num_merging_to, masking_ratio=mask_ratio, output_head=args.output_head)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos, num_merging_to=num_merging_to, masking_ratio=mask_ratio, output_head=args.output_head)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, args, ds=False, bf16=False, test_time=0):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    # args.eval_masking_ratio = random.uniform(args.eval_masking_ratio / 3, args.eval_masking_ratio)
    current_tokens = args.eval_true_frame * ((args.eval_input_size // 14) ** 2)

    if current_tokens <= args.eval_max_token_num:
        mask_ratio = None
        nums_token = current_tokens
    else:
        mask_ratio = 1 - args.eval_max_token_num / current_tokens
        nums_token = args.eval_max_token_num

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos, masking_ratio=mask_ratio, output_head=args.output_head)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos, masking_ratio=mask_ratio, output_head=args.output_head)
                loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].float().cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split(' ')[0]
            label = line.split(']')[-1].split(' ')[1]
            chunk_nb = line.split(']')[-1].split(' ')[2]
            split_nb = line.split(']')[-1].split(' ')[3]
            data = np.fromstring(' '.join(line.split(' ')[1:]).split('[')[1].split(']')[0], dtype=np.float32, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
