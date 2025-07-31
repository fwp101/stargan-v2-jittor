"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import jittor as jt
import jittor.nn as nn


def normalize(x, eps=1e-10):
    return x * jt.rsqrt(jt.sum(x**2, dim=1, keepdims=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Jittor官方模型库
        from jittor.models import alexnet
        self.model = alexnet(pretrained=True)
        self.channels = [64, 192, 384, 256, 256]

    def execute(self, x):
        # 按照PyTorch LPIPS的AlexNet特征层输出
        fmaps = []
        # AlexNet的features层结构
        features = self.model.features
        for i, layer in enumerate(features):
            x = layer(x)
            # 按照ReLU后输出特征
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def execute(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = jt.array([-0.03, -0.088, -0.188]).reshape(1, 3, 1, 1)
        self.sigma = jt.array([0.458, 0.448, 0.450]).reshape(1, 3, 1, 1)

    def _load_lpips_weights(self):
        import numpy as np
        own_state_dict = self.state_dict()
        state_dict = np.load('metrics/lpips_weights_jittor.npz')
        for name in own_state_dict:
            if name in state_dict:
                own_state_dict[name] = jt.array(state_dict[name])

    def execute(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet.execute(x)
        y_fmaps = self.alexnet.execute(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += jt.mean(conv1x1.execute((x_fmap - y_fmap) ** 2))
        return lpips_value


def calculate_lpips_given_images(group_of_images):
    # group_of_images = [jt.randn(N, C, H, W) for _ in range(10)]
    lpips = LPIPS().eval()
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # 分批 pairwise 计算，避免一次性分配全部显存
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_value = lpips.execute(group_of_images[i], group_of_images[j])
            jt.sync_all()
            lpips_values.append(lpips_value.item())
            jt.gc()
    del group_of_images
    jt.gc()
    # 直接用 numpy 计算均值，避免 jt.stack 占用显存
    import numpy as np
    return float(np.mean(lpips_values))