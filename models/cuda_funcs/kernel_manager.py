import torch
from typing import List, Optional, Tuple
import os
import inspect
import math


try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
except:
    pass


class Kernel:
    def __init__(self, name:str):
        self.name:str = name
        self.code:str = None
        self.raw_kernel:cp.RawKernel = None

    def __call__(self, grid, block, args, **kwargs):
        self.raw_kernel(grid, block, args, **kwargs)

class KernelManager:
    #position_enc_forward:Kernel = None
    #position_enc_backward:Kernel = None
    real_enc_fwd:Kernel = None
    real_enc_bwd:Kernel = None
    real_enc_fwd_v2:Kernel = None
    real_enc_bwd_v2:Kernel = None
    #position_enc_proj_forward:Kernel = None
    #position_enc_proj_backward:Kernel = None
    real_enc_proj_fwd:Kernel = None
    real_enc_proj_bwd:Kernel = None
    real_enc_proj_fwd_v2:Kernel = None
    real_enc_proj_bwd_v2:Kernel = None
    reci_enc_fwd:Kernel = None
    reci_enc_bwd:Kernel = None
    reci_enc_fwd_v2:Kernel = None
    reci_enc_bwd_v2:Kernel = None
    fused_dpa_fwd:Kernel = None
    fused_dpa_fwd_v2:Kernel = None
    fused_dpa_fwd_v3:Kernel = None
    fused_dpa_bwd:Kernel = None
    fused_dpa_bwd_v2:Kernel = None
    fused_dpa_bwd_v3:Kernel = None
    fused_dpa_bwd_q:Kernel = None
    fused_dpa_bwd_q_v2:Kernel = None
    fused_dpa_bwd_q_v3:Kernel = None
    irregular_transpose:Kernel = None
    irregular_transpose_old:Kernel = None
    irregular_mean_fwd:Kernel = None
    minimum_distance:Kernel = None
    
    MAX_SYSTEM_SIZE:int = 320
    MAX_SYSTEM_SIZE_POW2:int = int(2**math.ceil(math.log2(MAX_SYSTEM_SIZE)))
    RUNNING_SUM_LEN:int = 8

    @staticmethod
    def get_kernel_names() -> List[str]:
        return [name for name, attr in inspect.getmembers(KernelManager) \
                if  not name.startswith("_") and \
                    not inspect.isfunction(attr) and \
                    KernelManager.__annotations__.get(name, None) == Kernel
                ]

    @staticmethod
    def get_kernel(name:str) -> Kernel:
        return KernelManager.__dict__[name]

    @staticmethod
    def set_kernel(name:str, kernel:Kernel):
        setattr(KernelManager, name, kernel)
        #KernelManager.__dict__[name] = kernel

src_dir = os.path.dirname(os.path.abspath(__file__))
for name in KernelManager.get_kernel_names():
    kernel = Kernel(name)
    with open(os.path.join(src_dir, f'../kernels/{kernel.name}.cu'), 'r') as f:
        kernel.code = f.read()
    KernelManager.set_kernel(name, kernel)

# kernels = [
#     'position_enc_forward',
#     'position_enc_backward',
#     'adaptive_real_forward',
#     'adaptive_real_backward',
#     'position_enc_proj_forward',
#     'position_enc_proj_backward',
#     'adaptive_real_proj_forward',
#     'adaptive_real_proj_backward',
#     'reciprocal_forward',
#     'reciprocal_backward',
#     'fused_dpa_fwd',
#     'fused_dpa_bwd',
#     'fused_dpa_bwd_q',
#     'irregular_transpose',
#     'irregular_transpose_old',
# ]

# kernels = { name: Kernel(name) for name in kernels }

# src_dir = os.path.dirname(os.path.abspath(__file__))
# for name in kernels:
#     kernel = kernels[name]
#     with open(os.path.join(src_dir, f'../kernels/{kernel.name}.cu'), 'r') as f:
#         kernel.code = f.read()

def compile_kernels(lattice_range:int, head_num:int, key_head_dim:int, value_pe_dim:int, value_head_dim:int, set_minimum_range:bool):
    constants_dict = {
        'LATTICE_RANGE': str(lattice_range),
        'THREAD_NUM': str(head_num),
        'HEAD_NUM': str(head_num),
        'VPE_DIM': str(value_pe_dim),
        'V_HEAD_DIM': str(value_head_dim),
        'K_HEAD_DIM': str(key_head_dim),
        'SKIP_OUTOF_RADIUS': '0',
        'MINIMUM_RANGE': str(lattice_range) if set_minimum_range else '0',
        'MAX_SYSTEM_SIZE_POW2': KernelManager.MAX_SYSTEM_SIZE_POW2,
        'MAX_SYSTEM_SIZE': KernelManager.MAX_SYSTEM_SIZE,
        'RUNNING_SUM_LEN': KernelManager.RUNNING_SUM_LEN,
    }
    def replace_constants(code:str):
        for key,val in constants_dict.items():
            code = code.replace(key, val if isinstance(val, str) else str(val))
        return code
    
    options = ('-dc', '--std=c++11')
    if torch.cuda.device_count() > 0:
        with cp.cuda.Device(0):
            for name in KernelManager.get_kernel_names():
                kernel = KernelManager.get_kernel(name)
                code = replace_constants(kernel.code)
                kernel.raw_kernel = cp.RawKernel(code, kernel.name, options, jitify=True)
