import torch
import jittor as jt
import numpy as np

pt_ckpt = torch.load('wing.ckpt', map_location='cpu')
pt_ckpt = pt_ckpt['state_dict']  # 只取参数部分

jt_ckpt = {}
for k, v in pt_ckpt.items():
    jt_ckpt[k] = jt.array(np.array(v.cpu()))
jt.save(jt_ckpt, 'wing_jittor.ckpt')