
CUPY_AVAILABLE=True

try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
    from torch.utils.dlpack import to_dlpack, from_dlpack
    ppe.cuda.use_torch_mempool_in_cupy()
except:
    CUPY_AVAILABLE = False

from .kernel_manager import Kernel, KernelManager, compile_kernels
from .real_space_enc import RealPeriodicEncodingFuncCUDA
from .real_space_enc_proj import RealPeriodicEncodingWithProjFuncCUDA
from .reci_space_enc import ReciPeriodicEncodingFuncCUDA
from .fused_dpa import FusedDotProductAttentionCUDA
from .irregular_mean import IrregularMeanCUDA

__all__ = [
    'KernelManager', 
    'Kernel',
    'compile_kernels',
    'FusedDotProductAttentionCUDA',
    'RealPeriodicEncodingFuncCUDA',
    'RealPeriodicEncodingWithProjFuncCUDA',
    'ReciPeriodicEncodingFuncCUDA',
    'IrregularMeanCUDA',
    'CUPY_AVAILABLE',
]
