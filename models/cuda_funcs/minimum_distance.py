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

def compute_minimum_distance(rpos_ij_e, tvecs_n, batch_i, edge_ij_e, rvlen_n, cutoff_radius):
    assert cutoff_radius >= 0
    E = rpos_ij_e.shape[0]
    dev = rpos_ij_e.device
    
    rpos_ij_e = rpos_ij_e.contiguous()
    tvecs_n = tvecs_n.contiguous()
    batch_i = batch_i.contiguous()
    edge_ij_e = edge_ij_e.contiguous()
    rvlen_n = rvlen_n.contiguous()

    dist2_min_e = torch.empty((E, ), device=dev, dtype=rpos_ij_e.dtype)
    
    bsz = 32
    with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
        KernelManager.minimum_distance( ((E+bsz-1)//bsz, ), (bsz, ), (
            _to_copy(rpos_ij_e),
            _to_copy(tvecs_n),
            _to_copy(batch_i),
            _to_copy(edge_ij_e),
            E,
            _to_copy(rvlen_n),
            cutoff_radius,
            _to_copy(dist2_min_e),
        ))
    return dist2_min_e