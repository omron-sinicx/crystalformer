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

class ReciPeriodicEncodingFuncCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_ik, kr_base_e, rvecs_n, vcell_n, batch_i, edge_ij_e):
        # a_ik      : (points, heads)
        # kr_base_e : (edges, 3)
        # rvecs_n   : (batch, 3, 3)
        # vcell_n   : (batch)
        # batch_i   : (points)
        # edge_ij_e : (2, edges)
        # z_ijk = log( sum_n exp( a_ik*|pj + t1*n1+t2*n2+t3*n3 - pi|^2 ) )
        #           : (edges, heads)
        N, H = a_ik.shape
        E = edge_ij_e.shape[1]
        kw = {'device': a_ik.device, 'dtype': a_ik.dtype}
        
        a_ik = a_ik.contiguous().detach()
        kr_base_e = kr_base_e.contiguous()
        rvecs_n = rvecs_n.contiguous()
        vcell_n = vcell_n.contiguous()
        batch_i = batch_i.contiguous()
        edge_ij_e = edge_ij_e.contiguous()

        z_ek = torch.empty((E, H), **kw)
        sumexp_ek = torch.empty((E, H), **kw)
        bsz = H
        dev = a_ik.device
        
        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            kernel = KernelManager.reci_enc_fwd_v2 if config.REPRODUCIBLITY_STATE >= 4 \
                else KernelManager.reci_enc_fwd
            kernel(((E*H+bsz-1)//bsz, ), (bsz, ),
                (
                    _to_copy(a_ik),
                    _to_copy(kr_base_e),
                    _to_copy(rvecs_n),
                    _to_copy(vcell_n),
                    _to_copy(batch_i),
                    _to_copy(edge_ij_e),
                    N, H, E,
                    _to_copy(z_ek),
                    _to_copy(sumexp_ek),
                )
            )

        ctx.save_for_backward(a_ik, kr_base_e, rvecs_n, vcell_n, batch_i, edge_ij_e, z_ek, sumexp_ek)
        return z_ek

    @staticmethod
    def backward(ctx, gz_ek):
        a_ik, kr_base_e, rvecs_n, vcell_n, batch_i, edge_ij_e, z_ek, sumexp_ek = ctx.saved_tensors
        N, H = a_ik.shape
        E = edge_ij_e.shape[1]

        e_start_i = torch.zeros(N+1, dtype=batch_i.dtype, device=batch_i.device)
        e_start_i.scatter_add_(0, edge_ij_e[0]+1, torch.ones_like(edge_ij_e[0]))
        e_start_i = e_start_i.cumsum(0)

        ga_ik = torch.empty_like(a_ik)
        bsz = H
        dev = a_ik.device
        
        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            from .. import global_config as config
            kernel = KernelManager.reci_enc_bwd_v2 if config.REPRODUCIBLITY_STATE >= 4 \
                else KernelManager.reci_enc_bwd
            kernel(((N*H+bsz-1)//bsz, ), (bsz, ),
                (
                    _to_copy(a_ik),
                    _to_copy(kr_base_e),
                    _to_copy(rvecs_n),
                    _to_copy(vcell_n),
                    _to_copy(batch_i),
                    _to_copy(edge_ij_e),
                    _to_copy(e_start_i),
                    _to_copy(z_ek.detach()),
                    _to_copy(gz_ek.detach().contiguous()),
                    _to_copy(sumexp_ek.detach()),
                    N, H, E,
                    _to_copy(ga_ik),
                )
            )
        return ga_ik, None, None, None, None, None
