import torch
from torch import Tensor
from .kernel_manager import KernelManager

try:
    import cupy as cp
    import pytorch_pfn_extras as ppe
    from torch.utils.dlpack import to_dlpack, from_dlpack
except:
    pass

def _to_copy(x):
    if x is not None:
        return cp.from_dlpack(to_dlpack(x))
    return 0

# This function can be implemented with scatter_add,
# which however does not ensure the reproducibility. 
class IrregularMeanCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_ik:Tensor, batch_i:Tensor, sizes:Tensor):
        # x         : (points, *)
        # batch_i   : (points)
        N = x_ik.shape[0]
        D = x_ik.numel() // N
        dev = x_ik.device
        kw = {'device': x_ik.device, 'dtype': x_ik.dtype}
        
        if sizes is None:
            B = batch_i.max().item()+1
            sizes = torch.zeros(B, dtype=torch.long, device=dev)
            sizes.scatter_add_(0, batch_i, torch.ones(batch_i.shape, dtype=torch.long, device=dev))
        else:
            B = sizes.shape[0]

        start_n = torch.constant_pad_nd(torch.cumsum(sizes, 0), (1,0))

        x_ik = x_ik.contiguous().detach()
        start_n = start_n.contiguous()

        o_nk = torch.empty((B, ) + x_ik.shape[1:], **kw)
        
        bsz = min(32, D)
        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            assert (sizes <= KernelManager.MAX_SYSTEM_SIZE_POW2).all(), "Increase MAX_SYSTEM_SIZE in KernelManager"
            
            KernelManager.irregular_mean_fwd(((B*D+bsz-1)//bsz, ), (bsz, ),
                (
                    _to_copy(x_ik),
                    _to_copy(start_n),
                    N, D,
                    _to_copy(o_nk),
                )
            )

        ctx.save_for_backward(batch_i, sizes)
        return o_nk

    @staticmethod
    def backward(ctx, go_nk):
        batch_i, sizes = ctx.saved_tensors
        shape = [go_nk.shape[0]] + [1 for _ in go_nk.shape[1:]]

        # This code matches the implmentation of torch.mean().
        # gx_ik = (go_nk * sizes.reshape(shape).float().reciprocal())[batch_i]
        
        gx_ik = (go_nk / sizes.reshape(shape).float())[batch_i]
        
        return gx_ik, None, None
