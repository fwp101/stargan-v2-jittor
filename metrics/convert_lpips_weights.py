import torch
import numpy as np

state_dict = torch.load('metrics/lpips_weights.ckpt', map_location='cpu')
np_state_dict = {}
for k, v in state_dict.items():
    np_state_dict[k] = v.cpu().numpy()
np.savez('metrics/lpips_weights_jittor.npz', **np_state_dict)
print('转换完成，已保存为 metrics/lpips_weights_jittor.npz')