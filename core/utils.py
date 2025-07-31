

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import jittor as jt
import jittor.nn as nn


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    """
    He初始化(Kaiming初始化)，适用于ReLU及其变体激活函数
    统一的网络初始化函数，适用于所有StarGAN-v2网络
    """
    if isinstance(module, nn.Conv2d):
        # 卷积层使用He初始化
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        # 线性层使用He初始化
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.ConvTranspose2d):
        # 转置卷积层使用He初始化
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        # 归一化层的参数初始化
        if hasattr(module, 'weight') and module.weight is not None:
            # 检查是否为Jittor变量
            if isinstance(module.weight, jt.Var):
                nn.init.constant_(module.weight, 1.0)
            else:
                # 对于非Jittor变量的情况，直接赋值
                module.weight = 1.0
        if hasattr(module, 'bias') and module.bias is not None:
            # 检查是否为Jittor变量
            if isinstance(module.bias, jt.Var):
                nn.init.constant_(module.bias, 0.0)
            else:
                # 对于非Jittor变量的情况，直接赋值
                module.bias = 0.0
    # 对于其他类型的层（如Embedding等），不进行初始化



def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    # Simple replacement for torchvision.utils.save_image
    import cv2
    
    x = denormalize(x)
    
    # Convert tensor to numpy
    if isinstance(x, jt.Var):
        x = x.detach().numpy()
    
    # Denormalize from [0, 1] to [0, 255]
    x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    
    # Rearrange dimensions from (N, C, H, W) to (N, H, W, C)
    if x.ndim == 4:
        x = x.transpose(0, 2, 3, 1)
        # Create grid
        n, h, w, c = x.shape
        cols = ncol
        rows = (n + cols - 1) // cols
        grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        
        for i in range(n):
            row = i // cols
            col = i % cols
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = x[i]
        
        # Save
        if c == 3:
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, grid)
        elif c == 1:
            cv2.imwrite(filename, grid.squeeze())
    else:
        # Single image
        if x.ndim == 3:
            x = x.transpose(1, 2, 0)
        if x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, x)
        else:
            cv2.imwrite(filename, x.squeeze())


@jt.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = jt.concat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@jt.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = jt.randn(10000, latent_dim)
        y_many = jt.full((10000,), y_trg[0].item(), dtype=jt.int64)
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = jt.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = s_avg * psi + s_trg * (1 - psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = jt.concat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@jt.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = jt.ones(1, C, H, W)
    x_src_with_wb = jt.concat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = jt.concat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = jt.concat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


@jt.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [jt.full((N,), y, dtype=jt.int64)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = jt.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = s_prev * (1 - alpha) + s_next * alpha  # 替换torch.lerp
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = jt.concat([x_src, x_fake], dim=2)
        # 简化make_grid功能
        frame = entries.unsqueeze(0)
        frames.append(frame)
    frames = jt.concat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = -jt.ones((T, C, H*2, W + margin))
    merged = jt.concat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@jt.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = jt.concat([slided, interpolated], dim=3)  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(jt.concat(video))
    save_video(fname, video)


@jt.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = jt.randn(10000, latent_dim)
        y_many = jt.full((10000,), y_trg[0].item(), dtype=jt.int64)
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = jt.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = s_avg * psi + s_trg * (1 - psi)  # 替换torch.lerp
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next)
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(jt.concat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = jt.clamp(images * 0.5 + 0.5, 0, 1)
    return images.numpy().transpose(0, 2, 3, 1) * 255