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

class FusedDotProductAttentionCUDA_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk, batch_i, edge_ij_e):
        N, H, K = que_ihk.shape
        E = edge_ij_e.shape[1]
        dev = que_ihk.device

        e_start_i = torch.zeros(N+1, dtype=batch_i.dtype, device=batch_i.device)
        e_start_i.scatter_add_(0, edge_ij_e[0]+1, torch.ones_like(edge_ij_e[0]))
        e_start_i = e_start_i.cumsum(0)

        que_ihk = que_ihk.contiguous().detach()
        key_ihk = key_ihk.contiguous().detach()
        val_ihk = val_ihk.contiguous().detach()
        aij_eh = aij_eh.contiguous().detach() if aij_eh is not None else None
        bij_ehk = bij_ehk.contiguous().detach() if bij_ehk is not None else None
        batch_i = batch_i.contiguous()
        edge_ij_e = edge_ij_e.contiguous()

        output = torch.empty_like(val_ihk)
        prob_eh = torch.empty((E, H), dtype=que_ihk.dtype, device=dev)
        
        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            bsz = 1
            KernelManager.fused_dpa_fwd_v2(
                ((N*H+bsz-1)//bsz,), (bsz,),
                (
                    _to_copy(que_ihk),
                    _to_copy(key_ihk),
                    _to_copy(val_ihk),
                    _to_copy(aij_eh),
                    _to_copy(bij_ehk),
                    _to_copy(edge_ij_e),
                    _to_copy(e_start_i),
                    N, H, E,
                    _to_copy(prob_eh),
                    _to_copy(output),
                )
            )
        
        ctx.save_for_backward(que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk, 
                              batch_i, edge_ij_e, e_start_i, 
                              prob_eh, output)
        return output

    @staticmethod
    def backward(ctx, go_ihk):
        que_ihk, key_ihk, val_ihk, aij_eh, bij_ehk, \
        batch_i, edge_ij_e, e_start_i, \
        prob_eh, output = ctx.saved_tensors

        N, H, K = que_ihk.shape
        E = edge_ij_e.shape[1]
        dev = que_ihk.device

        B = batch_i.max().item()+1
        sizes = torch.zeros(B, dtype=torch.long, device=dev)
        sizes.scatter_add_(0, batch_i, torch.ones_like(batch_i))
        sizes2 = sizes*sizes

        if False:
            gque = []
            gkey = []
            gval = []
            gbij = []
            gaij = []
            sizes = torch.zeros(N, dtype=batch_i.dtype, device=batch_i.device)
            sizes.scatter_add_(0, batch_i, torch.ones_like(batch_i))
            sizes2 = sizes*sizes
            _sizes = sizes.tolist()
            _sizes2 = sizes2.tolist()
            for q,k,v,a,b,o,p,go,s in zip(
                que_ihk.split_with_sizes(_sizes),
                key_ihk.split_with_sizes(_sizes),
                val_ihk.split_with_sizes(_sizes),
                aij_eh.split_with_sizes(_sizes2),
                bij_ehk.split_with_sizes(_sizes2),
                output.split_with_sizes(_sizes),
                prob_eh.split_with_sizes(_sizes2),
                go_ihk.split_with_sizes(_sizes),
                _sizes):
                # q/k/v/o/go: (S, H, K)
                # a/p: (S*S, H)
                # b: (S*S, H, K)
                gb = go.reshape(s,1,H,K) * p.reshape(s,s,H,1)
                gv = gb.sum(dim=0)
                gval.append(gv)
                gbij.append(gb.reshape(s*s,H,K))
                gsm = (v.reshape(1,s,H,K) + b.reshape(s,s,H,K) - o.reshape(s,1,H,K))*gb
                ga = gsm.sum(dim=3)
                gq = (ga.reshape(s,s,H,1)*k.reshape(1,s,H,K)).sum(dim=1)
                gk = (ga.reshape(s,s,H,1)*q.reshape(s,1,H,K)).sum(dim=0)
                gaij.append(ga.reshape(s*s,H))
                gque.append(gq)
                gkey.append(gk)

            gque = torch.cat(gque)
            gkey = torch.cat(gkey)
            gval = torch.cat(gval)
            gbij = torch.cat(gbij)
            gaij = torch.cat(gaij)
            return gque, gkey, gval, gaij, gbij, None, None

        gque = torch.empty_like(que_ihk)
        gkey = torch.empty_like(key_ihk)
        gval = torch.empty_like(val_ihk)
        gaij = torch.empty_like(aij_eh)
        gbij = torch.empty_like(bij_ehk) if bij_ehk is not None else None
        go_ihk = go_ihk.contiguous().detach()
        
        tprob_eh = torch.empty_like(prob_eh)
        tbij_ehk = torch.empty_like(bij_ehk) if bij_ehk is not None else None
        start_inds = torch.constant_pad_nd(sizes2.cumsum(0), (1,0))

        with cp.cuda.Device(dev.index), ppe.cuda.stream(torch.cuda.current_stream(dev)):
            upper_mask = edge_ij_e[0] <= edge_ij_e[1]
            hE = upper_mask.long().sum().item()
            upper_e_t = torch.arange(E, dtype=torch.long, device=dev)[upper_mask]
            upper_batch_t = batch_i[edge_ij_e[0, upper_mask]]
            mat_sec_t = start_inds[upper_batch_t]
            sizes_t = sizes[upper_batch_t]

            def irregular_transpose(src:Tensor, dst:Tensor, C:int):
                bsz = min(32, C)
                KernelManager.irregular_transpose(
                    ((hE*C+bsz-1)//bsz, ), (bsz, ),
                    (_to_copy(src), _to_copy(upper_e_t), _to_copy(mat_sec_t), _to_copy(sizes_t), hE, C, _to_copy(dst))
                )

            irregular_transpose(prob_eh, tprob_eh, H)
            
            if bij_ehk is not None:
                irregular_transpose(bij_ehk, tbij_ehk, H*K)

            assert (sizes <= 1024).all(), "Max system size is 1024"
            bsz = 1
            KernelManager.fused_dpa_bwd_v2(
                ((N*H+bsz-1)//bsz,), (bsz,),
                (
                    _to_copy(que_ihk),
                    _to_copy(val_ihk),
                    _to_copy(tbij_ehk),
                    _to_copy(edge_ij_e),
                    _to_copy(e_start_i),
                    N, H, E,
                    _to_copy(tprob_eh),
                    _to_copy(output),
                    _to_copy(go_ihk),
                    _to_copy(gkey),
                    _to_copy(gval),
                    _to_copy(gaij),
                    _to_copy(gbij),
                )
            )

            # tranpose gaij and gbij
            irregular_transpose(gaij, gaij, H)
            if gbij is not None:
                irregular_transpose(gbij, gbij, H*K)
            
            # use gaij as grad softmax to compute grad q.
            bsz = 1
            KernelManager.fused_dpa_bwd_q_v2(
                ((N*H+bsz-1)//bsz,), (bsz,),
                (
                    _to_copy(key_ihk),
                    _to_copy(gaij),
                    _to_copy(edge_ij_e),
                    _to_copy(e_start_i),
                    N, H, E,
                    _to_copy(gque),
                )
            )

        return gque, gkey, gval, gaij, gbij, None, None
