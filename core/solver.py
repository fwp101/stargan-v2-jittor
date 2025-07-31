

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import jittor as jt
import jittor.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            
            # 创建统一的生成器优化器来解决Jittor参数隔离问题
            all_generator_params = (
                list(self.nets.mapping_network.parameters()) + 
                list(self.nets.generator.parameters()) + 
                list(self.nets.style_encoder.parameters())
            )
            
            # 统一生成器优化器
            self.optims.unified_generator = jt.optim.Adam(
                params=all_generator_params,
                lr=args.lr,
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay)
            
            # 判别器优化器
            self.optims.discriminator = jt.optim.Adam(
                params=self.nets.discriminator.parameters(),
                lr=args.lr,
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay)

            # 保持兼容性的单独优化器（现在不使用，但保留以备用）
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = jt.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=False, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=False, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=False, **self.nets_ema)]

        # 添加损失记录用于绘制曲线
        if args.mode == 'train':
            self.loss_history = {
                'steps': [],
                # 判别器损失 - 分别记录latent和ref模式
                'D_real_latent': [], 'D_fake_latent': [], 'D_reg_latent': [],
                'D_real_ref': [], 'D_fake_ref': [], 'D_reg_ref': [],
                # 生成器损失 - 分别记录latent和ref模式
                'G_adv_latent': [], 'G_sty_latent': [], 'G_ds_latent': [], 'G_cyc_latent': [],
                'G_adv_ref': [], 'G_sty_ref': [], 'G_ds_ref': [], 'G_cyc_ref': [],
                'lambda_ds': []
            }

        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                
                # 统一使用He初始化，适用于所有网络
                network.apply(utils.he_init)
                print(f'  - Applied He initialization for {name}')
                
                # 初始化后检查输出范围（可选，用于调试）
                if name == 'mapping_network':
                    with jt.no_grad():
                        # 测试映射网络输出范围
                        test_z = jt.randn(4, self.args.latent_dim)
                        test_y = jt.randint(0, self.args.num_domains, (4,))
                        test_output = network(test_z, test_y)
                        print(f'  - Output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]')
                        print(f'  - Output std: {test_output.std().item():.4f}')
                        
                elif name == 'style_encoder':
                    with jt.no_grad():
                        # 测试样式编码器输出范围
                        test_x = jt.randn(4, 3, self.args.img_size, self.args.img_size)
                        test_y = jt.randint(0, self.args.num_domains, (4,))
                        test_output = network(test_x, test_y)
                        print(f'  - Output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]')
                        print(f'  - Output std: {test_output.std().item():.4f}')
        

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()
    

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator - 按照Jittor示例控制梯度
            requires_grad(nets.generator, False)
            requires_grad(nets.mapping_network, False) 
            requires_grad(nets.style_encoder, False)
            requires_grad(nets.discriminator, True)
            
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self.optims.discriminator.zero_grad()
            self.optims.discriminator.backward(d_loss)
            self.optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self.optims.discriminator.zero_grad()
            self.optims.discriminator.backward(d_loss)
            self.optims.discriminator.step()

            # train the generator - 统一的生成器训练阶段
            # 修改后的训练循环（generator部分）
            requires_grad(nets.discriminator, False)
            requires_grad(nets.generator, True)
            requires_grad(nets.mapping_network, True)
            requires_grad(nets.style_encoder, True)

            # 分别计算各网络的损失
            g_loss_latent, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            g_loss_ref, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)

            # 独立更新各网络 - 修复：正确的Jittor优化器使用方式
            self._reset_grad()
            total_loss = g_loss_latent + g_loss_ref
            
            # 确保损失是Jittor变量而非numpy数组
            if not isinstance(total_loss, jt.Var):
                print(f"⚠️  警告: 损失不是Jittor变量，类型为 {type(total_loss)}")
                total_loss = jt.array(total_loss)
            
            # 分别计算各网络的梯度 - 修复：使用正确的Jittor反向传播方式
            self._reset_grad()
            total_loss = g_loss_latent + g_loss_ref
            
            # 确保损失是Jittor变量而非numpy数组
            if not isinstance(total_loss, jt.Var):
                print(f"⚠️  警告: 损失不是Jittor变量，类型为 {type(total_loss)}")
                total_loss = jt.array(total_loss)
            
            # 使用统一生成器优化器进行反向传播 - 解决Jittor参数隔离问题
            self.optims.unified_generator.zero_grad()
            self.optims.unified_generator.backward(total_loss)
            self.optims.unified_generator.step()
            

            # compute moving average of network parameters - 使用Jittor示例的accumulate函数
            accumulate(nets_ema.generator, nets.generator)
            accumulate(nets_ema.mapping_network, nets.mapping_network)
            accumulate(nets_ema.style_encoder, nets.style_encoder)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)
            

            # print out log info
            if (i+1) % args.print_every == 0:
                # 记录损失值用于绘制曲线 - 分别记录latent和ref模式
                self.loss_history['steps'].append(i+1)
                # 判别器损失 - latent模式
                self.loss_history['D_real_latent'].append(d_losses_latent.real)
                self.loss_history['D_fake_latent'].append(d_losses_latent.fake)
                self.loss_history['D_reg_latent'].append(d_losses_latent.reg)
                # 判别器损失 - ref模式
                self.loss_history['D_real_ref'].append(d_losses_ref.real)
                self.loss_history['D_fake_ref'].append(d_losses_ref.fake)
                self.loss_history['D_reg_ref'].append(d_losses_ref.reg)
                # 生成器损失 - latent模式
                self.loss_history['G_adv_latent'].append(g_losses_latent.adv)
                self.loss_history['G_sty_latent'].append(g_losses_latent.sty)
                self.loss_history['G_ds_latent'].append(g_losses_latent.ds)
                self.loss_history['G_cyc_latent'].append(g_losses_latent.cyc)
                # 生成器损失 - ref模式
                self.loss_history['G_adv_ref'].append(g_losses_ref.adv)
                self.loss_history['G_sty_ref'].append(g_losses_ref.sty)
                self.loss_history['G_ds_ref'].append(g_losses_ref.ds)
                self.loss_history['G_cyc_ref'].append(g_losses_ref.cyc)
                # 多样性损失权重
                self.loss_history['lambda_ds'].append(args.lambda_ds)
                
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # plot loss curves every 10000 steps
            if (i+1) % 10000 == 0:
                print(f"Plotting loss curves at step {i+1}...")
                self.plot_loss_curves(i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')
        
        # 训练结束时绘制最终的损失曲线
        print("Training completed. Plotting final loss curves...")
        self.plot_loss_curves(args.total_iters)

    @jt.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @jt.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

    def plot_loss_curves(self, step):
        """绘制从0步到当前步的各损失曲线并保存 - 分别显示latent和ref模式"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.loss_history['steps']:
                print("No loss history to plot")
                return
            
            # 创建保存目录
            os.makedirs(self.args.sample_dir, exist_ok=True)
            
            # 创建包含所有损失的大图 - 3x3布局
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle(f'Loss Curves: Latent vs Reference (Step 0 - {step})', fontsize=16)
            
            steps = np.array(self.loss_history['steps'])
            
            # 判别器损失对比
            axes[0, 0].plot(steps, self.loss_history['D_real_latent'], label='D_real (latent)', color='red', alpha=0.7)
            axes[0, 0].plot(steps, self.loss_history['D_real_ref'], label='D_real (ref)', color='darkred', alpha=0.7, linestyle='--')
            axes[0, 0].set_title('Discriminator Real Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(steps, self.loss_history['D_fake_latent'], label='D_fake (latent)', color='blue', alpha=0.7)
            axes[0, 1].plot(steps, self.loss_history['D_fake_ref'], label='D_fake (ref)', color='darkblue', alpha=0.7, linestyle='--')
            axes[0, 1].set_title('Discriminator Fake Loss')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].plot(steps, self.loss_history['D_reg_latent'], label='D_reg (latent)', color='green', alpha=0.7)
            axes[0, 2].plot(steps, self.loss_history['D_reg_ref'], label='D_reg (ref)', color='darkgreen', alpha=0.7, linestyle='--')
            axes[0, 2].set_title('Discriminator Regularization Loss')
            axes[0, 2].set_xlabel('Steps')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 生成器损失对比
            axes[1, 0].plot(steps, self.loss_history['G_adv_latent'], label='G_adv (latent)', color='orange', alpha=0.7)
            axes[1, 0].plot(steps, self.loss_history['G_adv_ref'], label='G_adv (ref)', color='darkorange', alpha=0.7, linestyle='--')
            axes[1, 0].set_title('Generator Adversarial Loss')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(steps, self.loss_history['G_sty_latent'], label='G_sty (latent)', color='purple', alpha=0.7)
            axes[1, 1].plot(steps, self.loss_history['G_sty_ref'], label='G_sty (ref)', color='indigo', alpha=0.7, linestyle='--')
            axes[1, 1].set_title('Style Reconstruction Loss')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(steps, self.loss_history['G_ds_latent'], label='G_ds (latent)', color='brown', alpha=0.7)
            axes[1, 2].plot(steps, self.loss_history['G_ds_ref'], label='G_ds (ref)', color='saddlebrown', alpha=0.7, linestyle='--')
            axes[1, 2].set_title('Diversity Sensitive Loss')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            # 循环一致性损失和lambda_ds
            axes[2, 0].plot(steps, self.loss_history['G_cyc_latent'], label='G_cyc (latent)', color='cyan', alpha=0.7)
            axes[2, 0].plot(steps, self.loss_history['G_cyc_ref'], label='G_cyc (ref)', color='darkcyan', alpha=0.7, linestyle='--')
            axes[2, 0].set_title('Cycle Consistency Loss')
            axes[2, 0].set_xlabel('Steps')
            axes[2, 0].set_ylabel('Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Lambda DS权重变化
            axes[2, 1].plot(steps, self.loss_history['lambda_ds'], label='lambda_ds', color='magenta', alpha=0.7)
            axes[2, 1].set_title('Diversity Loss Weight (lambda_ds)')
            axes[2, 1].set_xlabel('Steps')
            axes[2, 1].set_ylabel('Weight')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            # 总损失对比 - latent vs ref
            d_total_latent = (np.array(self.loss_history['D_real_latent']) + 
                             np.array(self.loss_history['D_fake_latent']) + 
                             np.array(self.loss_history['D_reg_latent']))
            d_total_ref = (np.array(self.loss_history['D_real_ref']) + 
                          np.array(self.loss_history['D_fake_ref']) + 
                          np.array(self.loss_history['D_reg_ref']))
            g_total_latent = (np.array(self.loss_history['G_adv_latent']) + 
                             np.array(self.loss_history['G_sty_latent']) + 
                             np.array(self.loss_history['G_cyc_latent']) - 
                             np.array(self.loss_history['G_ds_latent']))
            g_total_ref = (np.array(self.loss_history['G_adv_ref']) + 
                          np.array(self.loss_history['G_sty_ref']) + 
                          np.array(self.loss_history['G_cyc_ref']) - 
                          np.array(self.loss_history['G_ds_ref']))
            
            axes[2, 2].plot(steps, d_total_latent, label='Total D (latent)', color='red', alpha=0.7, linewidth=2)
            axes[2, 2].plot(steps, d_total_ref, label='Total D (ref)', color='darkred', alpha=0.7, linewidth=2, linestyle='--')
            axes[2, 2].plot(steps, g_total_latent, label='Total G (latent)', color='blue', alpha=0.7, linewidth=2)
            axes[2, 2].plot(steps, g_total_ref, label='Total G (ref)', color='darkblue', alpha=0.7, linewidth=2, linestyle='--')
            axes[2, 2].set_title('Total Losses: Latent vs Reference')
            axes[2, 2].set_xlabel('Steps')
            axes[2, 2].set_ylabel('Loss')
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            loss_curve_path = ospj(self.args.sample_dir, f'{step:06d}_loss_curves_comparison.png')
            plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Loss curves (latent vs ref) saved to: {loss_curve_path}")
            
            # 额外保存一个简化的总损失对比图
            fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ax.plot(steps, d_total_latent, label='Total D Loss (latent)', color='red', alpha=0.8, linewidth=2)
            ax.plot(steps, d_total_ref, label='Total D Loss (ref)', color='darkred', alpha=0.8, linewidth=2, linestyle='--')
            ax.plot(steps, g_total_latent, label='Total G Loss (latent)', color='blue', alpha=0.8, linewidth=2)
            ax.plot(steps, g_total_ref, label='Total G Loss (ref)', color='darkblue', alpha=0.8, linewidth=2, linestyle='--')
            ax.set_title(f'Total Losses Comparison: Latent vs Reference (Step 0 - {step})')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            total_loss_path = ospj(self.args.sample_dir, f'{step:06d}_total_losses_comparison.png')
            plt.savefig(total_loss_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Total loss comparison (latent vs ref) saved to: {total_loss_path}")
            
        except ImportError:
            print("Warning: matplotlib not available, skipping loss curve plotting")
        except Exception as e:
            print(f"Error plotting loss curves: {e}")

    

    # ...existing code...
def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images - 按照Jittor示例的方式设置requires_grad
    x_real.requires_grad = True
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    
    # 使用Jittor示例中的R1正则化实现方式
    grad_real = jt.grad(out.sum(), x_real)
    grad_penalty = (
        grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()
    loss_reg = 0.5 * grad_penalty

    # with fake images
    # 注意：在Jittor中，no_grad会完全禁用梯度，无法重新启用
    if z_trg is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:  # x_ref is not None
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss - 确保完整的梯度流
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = jt.mean(jt.abs(s_pred - s_trg))

    # diversity sensitive loss - 修复计算图断裂问题
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    # 保持完整的梯度流
    loss_ds = jt.mean(jt.abs(x_fake - x_fake2))

    # cycle-consistency loss - 确保使用正确的masks
    if args.w_hpf > 0:
        cycle_masks = nets.fan.get_heatmap(x_fake)
    else:
        cycle_masks = None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=cycle_masks)
    loss_cyc = jt.mean(jt.abs(x_rec - x_real))

    # 确保所有损失项都是Jittor变量
    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    
    # 验证损失是否为有效的Jittor变量
    if not isinstance(loss, jt.Var):
        raise ValueError(f"计算损失不是Jittor变量，类型为: {type(loss)}")
    
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = beta * param_test.data + (1 - beta) * param.data


def requires_grad(model, flag=True):
    """设置模型参数是否需要梯度，按照Jittor示例的方式"""
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    """指数移动平均更新，最简化版本避免CUDA参数溢出"""
    # 直接赋值，避免复杂的融合操作
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        # 一步计算，减少中间变量和操作融合
        new_val = p1.data * decay + p2.data * (1 - decay)
        p1.data = new_val


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = jt.full_like(logits, target)
    # 直接使用稳定的binary_cross_entropy_with_logits公式
    # BCE_with_logits(x, y) = max(x, 0) - x * y + log(1 + exp(-abs(x)))
    max_val = jt.maximum(logits, 0)  # 使用maximum替代clamp
    loss = max_val - logits * targets + jt.log(1 + jt.exp(-jt.abs(logits)))
    return jt.mean(loss)


def r1_reg(d_out, x_in):
    """简化的R1正则化函数，主要逻辑已移到compute_d_loss中"""
    # 这个函数保留是为了向后兼容，实际上现在在compute_d_loss中直接实现
    grad_real = jt.grad(d_out.sum(), x_in)
    grad_penalty = (
        grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()
    reg_loss = 0.5 * grad_penalty
    return reg_loss








