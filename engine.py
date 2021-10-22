# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import utils
from sklearn.metrics.cluster import normalized_mutual_info_score
from evaluate_cluster import evaluate as eval_pred
from objective import KL, CE, HE, EH, ConfidenceBasedCE
from datasets import ImageNet, ImageNetLMDB
from itertools import product
import numpy as np
import time
from PIL import Image
from torchvision import transforms
import os

def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch, set_training_mode=True, scaler=None, logfn=None, wd_schedule=None, qt_schedule=None, teacher_model=None, momentum_schedule=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", fn=logfn)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lrb', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('wd', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('wdb', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    if scaler and args.crops_interact_style not in ['label', 'self_label']: # output the scale factor (if use mix precision) to watch the training stability.
        metric_logger.add_meter('scale', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    real_labels, pred_labels = [], []
    print_freq, iteration = 10, 0

    for imgs, real_label, img_index in metric_logger.log_every(data_loader, print_freq, header):
        utils.adjust_learning_rate(args, optimizer, data_loader, epoch*len(data_loader)+iteration)
        for i_pg, param_group in enumerate(optimizer.param_groups):
            if i_pg == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[epoch*len(data_loader) + iteration]

        optimizer.zero_grad()
        imgs = [im.to(device, non_blocking=True) for im in imgs]

        with torch.cuda.amp.autocast(enabled=(True if scaler else False)):
            # feat should be (gc+lc)B x dim
            feat = model(imgs)
            if args.aug != 'multicrop':
                args.local_crops_number = 0
            all_feats = feat.chunk(2+args.local_crops_number)
            all_probs = [torch.nn.functional.softmax(f/args.tau, dim=-1) for f in all_feats]
            n_views = len(all_feats)

            if args.use_momentum_encoder and args.crops_interact_style == 'sparse' and epoch < args.mme_epochs:
                teacher_output = teacher_model(imgs[:2]) # 2B * dim
                teacher_feats = teacher_output.chunk(2)
                teacher_probs = [torch.nn.functional.softmax(f/args.tau, dim=-1) for f in teacher_feats]
                loss = {'kl':0, 'eh':0, 'he':0}
                count = 0
                for i_teacher in range(len(teacher_probs)):
                    for i_student in range(len(all_probs)):
                        if i_teacher == i_student:
                            continue
                        loss['kl'] = loss['kl'] + KL(teacher_probs[i_teacher], all_probs[i_student], args)
                        count += 1
                loss['kl'] = loss['kl']/count

                count = 0
                #global_weight, local_weight, total_weight = 1, 2/(len(all_probs)-2), 0
                global_weight, local_weight, total_weight = 1, 1, 0
                for i_student in range(len(all_probs)):
                    current_weight = [global_weight, local_weight][i_student>1]
                    loss['eh'] = loss['eh'] + EH(all_probs[i_student], args) * current_weight
                    loss['he'] = loss['he'] + HE(all_probs[i_student], args) * current_weight
                    total_weight += current_weight
                loss['eh'] = loss['eh']/total_weight
                loss['he'] = loss['he']/total_weight

                loss['final'] = loss['kl'] + (1+args.lam1)*loss['eh'] - args.lam2*loss['he']

            if (not args.use_momentum_encoder) and args.crops_interact_style == 'sparse' and epoch < args.mme_epochs:
                loss, all_loss = {}, []
                for i1 in range(2):
                    for i2 in range(i1+1, n_views):
                        all_loss.append(criterion(all_feats[i1], all_feats[i2], use_queue=False))
                for k in all_loss[0].keys():
                    loss[k] = sum([single_loss[k]/len(all_loss) for single_loss in all_loss])

            if args.crops_interact_style == 'self_label' or (args.crops_interact_style == 'sparse' and epoch >= args.mme_epochs):
                loss = {'final': 0}
                # showing statistics
                if args.use_momentum_encoder:
                    teacher_output = teacher_model(imgs[:2]) # 2B * dim
                    teacher_feats = teacher_output.chunk(2)
                    teacher_probs = [torch.nn.functional.softmax(f/args.tau, dim=-1) for f in teacher_feats]
                    all_max = torch.stack([teacher_probs[0].max(dim=1)[0], teacher_probs[1].max(dim=1)[0]], dim=1).max(dim=1)[0]
                    all_max = all_max.detach()
                else:
                    all_max = torch.stack([all_probs[0].max(dim=1)[0], all_probs[1].max(dim=1)[0]], dim=1).max(dim=1)[0]
                    all_max = all_max.detach()

                if args.crops_interact_style == 'self_label':
                    quantile = qt_schedule[epoch*len(data_loader) + iteration]
                else:
                    quantile = qt_schedule[(epoch-args.mme_epochs)*len(data_loader) + iteration]

                if quantile != 0:
                    all_max = all_max.sort(descending=True)[0]
                    sup_num = int(quantile * all_max.size(0))
                    args.threshold = all_max[min(sup_num, all_max.size(0)-1)]
                    metric_logger.update(threshold=args.threshold)
                # all_max: [ bs ]

                frac = ((all_max > args.threshold).sum()/all_max.size(0)).item()
                metric_logger.update(frac=frac)

                criterion = ConfidenceBasedCE(args.threshold, apply_class_balancing=False)
                if args.use_momentum_encoder:
                    condition = (teacher_probs[0].max(dim=1)[0]>teacher_probs[1].max(dim=1)[0]).reshape(teacher_probs[0].size(0), 1).expand(teacher_probs[0].size(0), teacher_probs[1].size(1))
                    weak_anchor = torch.where(condition, teacher_feats[0], teacher_feats[1])
                else:
                    condition = (all_probs[0].max(dim=1)[0]>all_probs[1].max(dim=1)[0]).reshape(all_probs[0].size(0), 1).expand(all_probs[0].size(0), all_probs[1].size(1))
                    weak_anchor = torch.where(condition, all_feats[0], all_feats[1])

                strong_anchor = torch.where(condition, all_feats[1], all_feats[0])
                loss['final'] = criterion(weak_anchor, strong_anchor)
                for i_v in range(2, n_views):
                    loss['final'] = loss['final'] + criterion(weak_anchor, all_feats[i_v])
                loss['final'] = loss['final'] / (n_views - 1)

                loss['eh'] = 0.5 * (EH(all_probs[0], args) + EH(all_probs[1], args))
                loss['he'] = 0.5 * (HE(all_probs[0], args) + HE(all_probs[1], args))

        pred1 = utils.concat_all_gather(all_probs[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(all_probs[1].max(dim=1)[1]) 
        acc = (pred1 == pred2).sum()/pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(real_label.to(device).long()))

        if scaler:
            scaler.scale(loss['final']).backward()
            scale_value = scaler.get_scale()
            scaler.unscale_(optimizer)
            if args.clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss['final'].backward()
            optimizer.step()
        
        metric_logger.update(fc_std=model.module.projection_heads.f_column_std)
        metric_logger.update(fr_std=model.module.projection_heads.f_row_std)
        metric_logger.update(ftc_std=model.module.projection_heads.ft_column_std)
        metric_logger.update(ftr_std=model.module.projection_heads.ft_row_std)
        metric_logger.update(f_gn=model.module.projection_heads.gn_f)
        metric_logger.update(ft_gn=model.module.projection_heads.gn_ft)

        if args.use_momentum_encoder:
            with torch.no_grad():
                m = momentum_schedule[epoch*len(data_loader) + iteration]
                for param_q, param_k in zip(model.module.parameters(), teacher_model.module.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        for k, v in loss.items():
            if k == 'he':
                carl = utils.get_world_size() * math.log(utils.get_world_size())
                metric_logger.update(**{k:(v.item() + carl)/utils.get_world_size()})
            elif k == 'final':
                if args.crops_interact_style not in ['self_label',] and not (args.crops_interact_style == 'sparse' and epoch >= args.mme_epochs):
                    carl = - args.lam2 * utils.get_world_size() * math.log(utils.get_world_size())
                    metric_logger.update(**{k:(v.item() + carl)/utils.get_world_size()})
                else:
                    metric_logger.update(**{k:v.item()/utils.get_world_size()})
            else:
                metric_logger.update(**{k:v.item()/utils.get_world_size()})

        metric_logger.update(acc=acc)
        if scaler and args.crops_interact_style not in ['label', 'self_label']:
            metric_logger.update(scale=scale_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lrb=optimizer.param_groups[1]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(wdb=optimizer.param_groups[1]["weight_decay"])
        iteration = iteration + 1
    
    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dic = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.enable_watch or (epoch + 1) % 10 == 0:
        start_time_evalcluster = time.time()
        nmi, ami, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
        print("NMI: {}, AMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ami, ari, fscore, adjacc))
        return_dic.update({"nmi": nmi, "ami": ami, "ari": ari, "fscore": fscore, "adjacc": adjacc})
        end_time_evalcluster = time.time()
        print("calculating clustering indicators {}".format(end_time_evalcluster-start_time_evalcluster))
    return return_dic

@torch.no_grad()
def eval_one_epoch(args, model, data_loader, device, epoch=0, set_training_mode=False, scaler=None, logfn=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", fn=logfn)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    real_labels = []
    pred_labels = []
    all_img_idx = []
    total_probs = []
    for imgs, real_label, img_idx in metric_logger.log_every(data_loader, print_freq, header):
        imgs = imgs.to(device)
        with torch.cuda.amp.autocast(enabled=(True if scaler else False)):
            feats = model(imgs)
            all_probs = torch.nn.functional.softmax(feats/args.tau, dim=-1)

        pred = all_probs.max(dim=1)[1]
        pred = utils.concat_all_gather(pred) 
        pred_labels.append(pred)
        real_label_cat = utils.concat_all_gather(real_label.to(device).long())
        real_labels.append(real_label_cat)

        temp_probs = utils.concat_all_gather(all_probs)
        total_probs.append(temp_probs)

        img_idx_cat = utils.concat_all_gather(img_idx.to(device).long())
        all_img_idx.append(img_idx_cat)
        torch.cuda.synchronize()
    
    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    all_img_idx = torch.cat(all_img_idx).cpu().detach().numpy()
    total_probs = torch.cat(total_probs, dim=0).cpu().detach().numpy()

    ordered_real_labels = np.ones(len(all_img_idx))
    ordered_real_labels[all_img_idx] = real_labels
    ordered_pred_labels = np.ones(len(all_img_idx))
    ordered_pred_labels[all_img_idx] = pred_labels
    ordered_total_probs = np.ones_like(total_probs)
    ordered_total_probs[all_img_idx] = total_probs

    np.save('ordered_real_labels.npy', ordered_real_labels)
    np.save('ordered_pred_labels.npy', ordered_pred_labels)
    nmi, ami, ari, fscore, adjacc, image_match, mapped_preds, top5 = eval_pred(ordered_real_labels.astype(int), ordered_pred_labels.astype(int), calc_acc=(args.dim==1000), total_probs=ordered_total_probs)

    print("NMI: {}, AMI: {}, ARI: {}, F: {}, ACC: {}, ACC-Top5: {}".format(nmi, ami, ari, fscore, adjacc, top5))
    metric_logger.synchronize_between_processes()
    return_dic = {"nmi": nmi, "ami":ami, "ari": ari, "fscore": fscore, "adjacc": adjacc, "match":image_match, "mapped_preds": mapped_preds, "acc5": top5}
    return return_dic

@ torch.no_grad()
def inference(args, model, img_path, device, match):
    from imagenet1000_id_to_labels import id_to_labels
    model.train(False)
    aug = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.exists(img_path):
        dataset = ImageNetLMDB(args.data_path,'val.lmdb', aug)
        img, target, _ = dataset[int(img_path)]
    else:
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = aug(img).to(device).unsqueeze(0)
            target = None

    probs = torch.nn.functional.softmax(model(img), dim=-1)[0] # get a 1000 dimension vector
    probs, indices = probs.topk(10, dim=-1)
    probs = list(probs.cpu().detach().numpy())
    indices = list(indices.cpu().detach().numpy())
    pt_table = {}
    for pred_i, target_i in match:
        pt_table[pred_i] = target_i
    
    for i in range(len(probs)):
        p = probs[i]
        ind = indices[i]
        label = id_to_labels[pt_table[ind]]
        print(f'Top {i+1}: {p:.3f}  {label}')

    if target is not None:
        print(f'Target: {id_to_labels[target]}')

    if target is not None:
        return_dic = {
                'idx': int(img_path),
                'top_scores': probs,
                'label_ids': indices,
                'mapped_label_ids': [pt_table[i] for i in indices],
                'labels': [id_to_labels[pt_table[i]] for i in indices],
                'target': id_to_labels[target],
                'target_id': target,
        }
        torch.save(return_dic, f'prediction_{int(img_path)}.pth')
