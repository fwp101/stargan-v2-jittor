"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse


import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.models import inception_v3
from scipy import linalg
from core.data_loader import get_eval_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3_Jittor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True)
        # 取最后池化层输出
    def execute(self, x):
        # x: [N,3,H,W], range [0,1] or [-1,1]
        if x.min() < 0:
            x = (x + 1) / 2
        # Jittor官方模型直接 forward 得到 [N,1000] 分类特征
        out = self.model(x)
        return out.reshape(out.shape[0], -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)



def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    inception = InceptionV3_Jittor().eval()
    loaders = [get_eval_loader(path, img_size, batch_size)() for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader):
            # x: [N,3,H,W], jt.array
            actv = inception.execute(x)
            jt.sync_all()
            actvs.append(actv.numpy())
            jt.gc()
        if len(actvs) == 0:
            # 没有图片，返回 NaN
            return float('nan')
        actvs = np.concatenate(actvs, axis=0)
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return float(fid_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(args.paths, args.img_size, args.batch_size)
    print('FID: ', fid_value)