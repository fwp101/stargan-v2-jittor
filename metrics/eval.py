"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import jittor as jt  # [Jittor替换] 替换torch为jittor

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils


def calculate_metrics(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = 'cuda' if jt.has_cuda else 'cpu'  # [Jittor替换] torch.device->字符串

    # 只保留有图片的类别文件夹
    domains = [d for d in os.listdir(args.val_img_dir)
               if not d.startswith('.') and os.path.isdir(os.path.join(args.val_img_dir, d))]
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    lpips_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]

        if mode == 'reference':
            path_ref = os.path.join(args.val_img_dir, trg_domain)
            loader_ref = get_eval_loader(root=path_ref,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False,
                                         drop_last=True)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False)()

            task = '%s2%s' % (src_domain, trg_domain)
            # 禁止评估阶段保存图片，直接跳过相关目录操作
            path_fake = os.path.join(args.eval_dir, task)
            # pass

            # 跳过LPIPS计算，仅保留流程结构
            print('Skip LPIPS calculation for %s...' % task)
            lpips_dict['LPIPS_%s/%s' % (mode, task)] = None

        # delete dataloaders
        del loader_src
        if mode == 'reference':
            del loader_ref
            try:
                del iter_ref
            except NameError:
                pass

    # calculate the average LPIPS for all tasks
    # 跳过None值，避免报错
    valid_lpips = [v for v in lpips_dict.values() if v is not None]
    if valid_lpips:
        lpips_mean = float(np.mean(valid_lpips))
    else:
        lpips_mean = None
    lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    utils.save_json(lpips_dict, filename)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)


def calculate_fid_for_all_tasks(args, domains, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s2%s' % (src_domain, trg_domain)
            path_real = os.path.join(args.train_img_dir, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            # 这里不保存生成图片，只生成一遍用于FID计算
            # 你可以在 calculate_fid_given_paths 内部实现图片生成逻辑，但不要保存图片
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                img_size=args.img_size,
                batch_size=args.val_batch_size
            )
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils.save_json(fid_values, filename)
