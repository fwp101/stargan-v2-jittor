

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
import numpy as np
import cv2  # [Jittor替换] 用于图片读取
import jittor as jt  # [Jittor替换] 替换torch为jittor


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset:
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = str(self.samples[index])
        img = cv2.imread(fname)
        if img is None:
            raise RuntimeError(f'Failed to read image: {fname}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset:
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = [d for d in os.listdir(root) if not d.startswith('.')]
        fnames, fnames2, labels = [], [], []
        
        # 首先收集所有域的文件
        all_files_by_domain = {}
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            all_files_by_domain[idx] = cls_fnames
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        
        # 为每个文件生成第二个参考图像，确保来自不同域或不同图像
        for i, (fname, label) in enumerate(zip(fnames, labels)):
            # 70%概率从不同域采样，30%概率从同域不同图像采样
            if random.random() < 0.7 and len(domains) > 1:
                # 从不同域采样
                other_domains = [d for d in all_files_by_domain.keys() if d != label]
                if other_domains:
                    target_domain = random.choice(other_domains)
                    fname2 = random.choice(all_files_by_domain[target_domain])
                else:
                    # 如果只有一个域，从同域不同图像采样
                    domain_files = [f for f in all_files_by_domain[label] if f != fname]
                    fname2 = random.choice(domain_files) if domain_files else fname
            else:
                # 从同域不同图像采样
                domain_files = [f for f in all_files_by_domain[label] if f != fname]
                fname2 = random.choice(domain_files) if domain_files else fname
            
            fnames2.append(fname2)
        
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = cv2.imread(str(fname))
        img2 = cv2.imread(str(fname2))
        if img is None or img2 is None:
            raise RuntimeError(f'Failed to read image: {fname} or {fname2}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    """创建平衡采样器，与PyTorch版本保持一致"""
    if labels is None:
        return None
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    # 返回加权采样的indices
    indices = np.random.choice(len(labels), size=len(labels), replace=True, p=weights/weights.sum())
    return indices


def _simple_transform(img, img_size, prob=0.5):
    """简单的图像变换，与PyTorch版本保持一致"""
    # 随机裁剪（对应RandomResizedCrop）
    if random.random() < prob:
        h, w, _ = img.shape
        scale = random.uniform(0.8, 1.0)
        ratio = random.uniform(0.9, 1.1)
        crop_h = int(h * scale)
        crop_w = int(w * scale * ratio)
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        y = random.randint(0, max(1, h - crop_h))
        x = random.randint(0, max(1, w - crop_w))
        img = img[y:y+crop_h, x:x+crop_w]
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # 随机水平翻转
    if random.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
    
    # ToTensor + Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # 标准化到[-1, 1]
    
    # 确保创建的Jittor张量支持梯度计算
    img_tensor = jt.array(img).permute(2, 0, 1)
    # 对于训练数据，确保requires_grad为True（虽然输入数据通常不需要梯度，但确保兼容性）
    return img_tensor


class TrainDataLoader:
    """训练数据加载器，模拟PyTorch DataLoader的行为"""
    def __init__(self, dataset, batch_size, which, prob, img_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.which = which
        self.prob = prob
        self.img_size = img_size
        self.shuffle = shuffle
        
        # 获取采样indices
        if hasattr(dataset, 'targets') and dataset.targets is not None:
            self.indices = _make_balanced_sampler(dataset.targets)
        else:
            self.indices = np.arange(len(dataset))
            if shuffle:
                np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle and hasattr(self.dataset, 'targets') and self.dataset.targets is not None:
            # 重新生成平衡采样
            self.indices = _make_balanced_sampler(self.dataset.targets)
        elif self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        if len(batch_indices) == 0:
            raise StopIteration
        
        if self.which == 'reference':
            imgs, imgs2, labels = [], [], []
            for idx in batch_indices:
                if idx >= len(self.dataset):
                    continue
                img, img2, label = self.dataset[idx]
                # 确保img和img2是numpy数组
                if not isinstance(img, np.ndarray):
                    raise TypeError(f"Expected numpy array, got {type(img)}")
                if not isinstance(img2, np.ndarray):
                    raise TypeError(f"Expected numpy array, got {type(img2)}")
                
                imgs.append(_simple_transform(img, self.img_size, self.prob))
                imgs2.append(_simple_transform(img2, self.img_size, self.prob))
                labels.append(label)
            
            if len(imgs) == 0:
                raise StopIteration
                
            return jt.stack(imgs), jt.stack(imgs2), jt.array(labels)
        else:
            imgs, labels = [], []
            for idx in batch_indices:
                if idx >= len(self.dataset):
                    continue
                img = self.dataset[idx]
                # 确保img是numpy数组
                if not isinstance(img, np.ndarray):
                    raise TypeError(f"Expected numpy array, got {type(img)}")
                
                imgs.append(_simple_transform(img, self.img_size, self.prob))
                # 对于source数据，从文件路径推断label
                if hasattr(self.dataset, 'samples'):
                    # 简单的label推断，可以根据需要调整
                    labels.append(0)  # 默认label
                else:
                    labels.append(0)
            
            if len(imgs) == 0:
                raise StopIteration
                
            return jt.stack(imgs), jt.array(labels)


class EvalDataLoader:
    """评估数据加载器"""
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        if len(batch_indices) == 0:
            raise StopIteration
        
        imgs = []
        for idx in batch_indices:
            img = self.dataset[idx]
            imgs.append(img)
        
        return jt.stack(imgs)


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)
    
    if which == 'source':
        # 不应用transform，让TrainDataLoader中的_simple_transform处理
        dataset = DefaultDataset(root, transform=None)
    elif which == 'reference':
        # 不应用transform，让TrainDataLoader中的_simple_transform处理
        dataset = ReferenceDataset(root, transform=None)
    else:
        raise NotImplementedError
    
    return TrainDataLoader(dataset, batch_size, which, prob, img_size, shuffle=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    
    if imagenet_normalize:
        height, width = 299, 299
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        height, width = img_size, img_size
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])

    def eval_transform(img):
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.resize(img, (height, width))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        # 确保创建的Jittor张量正确
        img_tensor = jt.array(img).permute(2, 0, 1)
        return img_tensor

    dataset = DefaultDataset(root, transform=eval_transform)
    return EvalDataLoader(dataset, batch_size, shuffle)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    
    # 对于测试，我们仍然需要应用transform，但要确保逻辑正确
    def test_transform(img):
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        # 确保创建的Jittor张量正确
        img_tensor = jt.array(img).permute(2, 0, 1)
        return img_tensor

    dataset = DefaultDataset(root, transform=test_transform)
    
    # 为测试数据创建专门的加载器，不使用TrainDataLoader的_simple_transform
    class TestDataLoader:
        def __init__(self, dataset, batch_size, shuffle=True):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(len(dataset))
        
        def __iter__(self):
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0
            return self
        
        def __next__(self):
            if self.current_idx >= len(self.indices):
                raise StopIteration
            
            batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
            self.current_idx += self.batch_size
            
            if len(batch_indices) == 0:
                raise StopIteration
            
            imgs = []
            labels = []
            for idx in batch_indices:
                img = self.dataset[idx]  # 这里已经应用了test_transform
                imgs.append(img)
                labels.append(0)  # 默认label
            
            return jt.stack(imgs), jt.array(labels)
    
    return TestDataLoader(dataset, batch_size, shuffle)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            # [多样性增强] 确保z_trg和z_trg2有足够的差异
            z_trg = jt.randn(x.shape[0], self.latent_dim)
            z_trg2 = jt.randn(x.shape[0], self.latent_dim)
            
            # 确保两个随机向量之间有足够的距离，避免生成过于相似的样式
            # 如果距离太小，重新采样z_trg2
            for i in range(x.shape[0]):
                max_attempts = 5
                attempt = 0
                while attempt < max_attempts:
                    diff_norm = jt.norm(z_trg[i] - z_trg2[i]).item()
                    if diff_norm > 1.0:  # 确保欧几里得距离大于1.0
                        break
                    z_trg2[i] = jt.randn(self.latent_dim)
                    attempt += 1
            
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return inputs