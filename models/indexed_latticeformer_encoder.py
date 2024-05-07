
from dis import dis
from typing import List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from torch import Tensor, cos_
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn import Parameter, Module, Linear, Dropout, LayerNorm, ModuleList, Identity
import torch.nn.functional as F
from .indexed_lattice_multi_head_attention import IndexedLatticeMultiheadAttention
from .indexed_lattice_multi_head_attention_cuda import CrystalformerMultiheadAttentionCUDA
import math
from .latticeformer_params import LatticeformerParams

def get_edge_index(sizes, device):
    edges_i = []
    edges_j = []

    factory_kywd_long = {'dtype': torch.long, 'device': device}
    cur_index = 0
    sizes = sizes.tolist() if torch.is_tensor(sizes) else sizes
    for i,num in enumerate(sizes):
        inds = torch.arange(cur_index, cur_index+num, **factory_kywd_long)
        inds_i = inds.reshape(num,1).repeat(1, num).flatten()
        inds_j = inds.reshape(1,num).repeat(num, 1).flatten()
        edges_i.append(inds_i)
        edges_j.append(inds_j)
        cur_index += num
    edges_i = torch.cat(edges_i)
    edges_j = torch.cat(edges_j)
    edges = torch.stack((edges_i, edges_j))
    return edges.contiguous()

def det_3x3(m):
    #   0 1 2
    # 0 * * *
    # 1 * * *
    # 2 * * *
    return m[:, 0, 0]*m[:, 1, 1]*m[:, 2, 2] \
        + m[:, 0, 1]*m[:, 1, 2]*m[:, 2, 0] \
        + m[:, 0, 2]*m[:, 1, 0]*m[:, 2, 1] \
        - m[:, 0, 2]*m[:, 1, 1]*m[:, 2, 0] \
        - m[:, 0, 0]*m[:, 1, 2]*m[:, 2, 1] \
        - m[:, 0, 1]*m[:, 1, 0]*m[:, 2, 2]

def compute_lattice_distances(
    pos, batch, trans_vec, sizes, lattice_range,
    output_dis=True,
    output_rec=False,
    exclude_self=False,
    dim_pe_wave=0):
    # pos: (L,D)
    # batch: (L)
    # trans_vec: (N,D,D)
    # range: int

    D = pos.shape[-1]
    factory_kywd = {'dtype':pos.dtype, 'device': pos.device}
    factory_kywd_long = {'dtype':torch.long, 'device': pos.device}

    # split flat-batched data
    if sizes is None:
        sizes = torch.zeros(trans_vec.shape[0], **factory_kywd_long)
        sizes.scatter_add_(0, batch, torch.ones_like(batch))
    if torch.is_tensor(sizes):
        sizes = sizes.tolist()
    edges = get_edge_index(sizes, pos.device)
    
    # define the range of lattice to be considered
    grids = torch.arange(-lattice_range, lattice_range+1, **factory_kywd)
    grids = torch.stack(torch.meshgrid([grids]*D, indexing='ij'), dim=-1)
    # grids: (2*r+1,2*r+1,2*r+1,D)
    grids = grids.reshape(-1, D)
    # grids: ((2*R+1)^3, D) = (R, D)

    dis_pql = cos_kr = k2 = vcell = self_edges = None
    wk_pql = None

    # TODO: validate pe_wave: cos -> sin
    # vcell seems to be always positive for MP data.

    # Note torch.det sometimes yields nan when MAGMA solver is used.
    # To avoid it, call torch.backends.cuda.preferred_linalg_library("cusolver")
    # See also https://github.com/pytorch/pytorch/issues/73622 
    # vcell = torch.det(trans_vec)

    # Further note: there was still an error saying
    # torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED, when calling `cusolverDnSgetrf`. This error may appear if the input matrix contains NaN.
    # So, the det is now implemented manually.
    vcell = det_3x3(trans_vec)
    recip_vec = torch.cat([
        torch.cross(trans_vec[:, 1:2], trans_vec[:, 2:3], dim=2),
        torch.cross(trans_vec[:, 2:3], trans_vec[:, 0:1], dim=2),
        torch.cross(trans_vec[:, 0:1], trans_vec[:, 1:2], dim=2),
    ], dim=1)*(2.0*math.pi/vcell[:,None,None])

    b2e = batch[edges[0]]                                           # (N)[b2e] -> (E)
    pos_p_q = pos[edges[1]] - pos[edges[0]]                         # (E,D)
    pos_lat = grids @ trans_vec                                     # (N,R,D)   = (  R,D)x(N, D,D)
    pos_pql = pos_p_q[:,None] + pos_lat[b2e]                        # (E,R,D)   = (E, 1,D)+(E, R,D)
    del pos_lat

    if dim_pe_wave > 0:
        u = torch.arange(1, dim_pe_wave+1, **factory_kywd)
        u = (2.0*u - (dim_pe_wave+1)) / (2.0*dim_pe_wave)
        u = torch.stack(torch.broadcast_tensors(
            u[:,None,None],
            u[None,:,None],
            u[None,None,:]
        ), dim=-1).view(-1, 3)

        wk = u @ recip_vec                                              # (N,K,D)   = (  K,D)x(N, D,D)

        K = dim_pe_wave**3
        if False:
            wk_pql = torch.zeros(pos_pql.shape[:2]+(2*K,), **factory_kywd)

            # because wk_pql is large, compute it with minimum temporary memory usage
            wk = wk[b2e]
            
            wk_pql[..., :K] += pos_pql[:,:,None,0]*wk[:,None,:,0] 
            wk_pql[..., :K] += pos_pql[:,:,None,1]*wk[:,None,:,1]
            wk_pql[..., :K] += pos_pql[:,:,None,2]*wk[:,None,:,2]
            wk_pql[..., K:] = wk_pql[..., :K]
            # wk_pql[..., :K] = pos_pql @ wk.transpose(1,2)[b2e]            # (E,R,K) = (E,R,D) x (E,D,K)
            # wk_pql[..., K:] = wk_pql[..., :K]

            torch.cos_(wk_pql[:,:, :K])
            torch.sin_(wk_pql[:,:, K:])
        else:
            wk_pql = pos_pql @ wk.transpose(1,2)[b2e]
            torch.sin_(wk_pql)
        del wk, u

    if output_dis:
        dis_pql = torch.norm(pos_pql,dim=-1)                        # (E,R)
    
    if exclude_self:
        s2 = [s*s for s in sizes]
        c2 = torch.tensor([0]+s2[:-1], **factory_kywd_long).cumsum(0).tolist()
        self_edges = torch.cat([torch.arange(s,**factory_kywd_long)*(1+s)+c for s,c in zip(sizes,c2)])
        if dis_pql is not None:
            dis_pql[self_edges, dis_pql.shape[1]//2] = 1e8

    if output_rec:
        half_grids = grids[grids.shape[0]//2:]
        assert abs(half_grids + grids[0:grids.shape[0]//2+1].flip(0)).sum() == 0

        # [<r,b1>, <r,b2>, <r,b3>]
        cos_kr = pos_p_q[:,None] @ recip_vec.transpose(1,2)[b2e]    # (E,1,D)   = (E, 1,D)x(E, D,D)
        # <k,r> = n1*<r,b1> + n2*<r,b2> + n3*<r,b3>]
        cos_kr.squeeze_(1)                                          # (E,D)
        cos_kr = half_grids @ cos_kr[...,None]                      # (E,R,1)   = (  R,D)x(E, D,1)
        cos_kr.squeeze_(2)                                          # (E,R)
        cos_kr.cos_()
        cos_kr[:,1:] += cos_kr[:,1:]

        k2 = recip_vec @ recip_vec.transpose(1,2)        # (N,D,D)
        h = half_grids[:,:,None] @ half_grids[:,None,:]  # (R,D,D) = (R,D,1)x(R,1,D)
        k2 = (k2[:,None]*h[None]).sum(dim=(2,3))         # (N,R)   = (N,R,D,D).sum(DD)
        del h

        # When torch.backends.cuda.matmul.allow_tf32 = True,
        # the computations of cos_kr and k2 become very sensitive to numerical errors.
        # The above code produced more accurate values than the code below.
        # when compared to double-based computations as reference.
        if False:
            # k = n1*b1 + n2*b2 + n3*b3
            k2 = half_grids @ recip_vec                                 # (N,R,D)   = (   R,D)x(N, D,D)

            cos_kr = k2[b2e] @ pos_p_q[...,None]                        # (E,R,1)   = (E, R,D)x(E, D,1)
            cos_kr.squeeze_(2)                                          # (E,R)
            cos_kr.cos_()
            cos_kr[:,1:] += cos_kr[:,1:]

            # k2 = <k,k>
            k2 = k2[...,None,:] @ k2[...,:,None]                        # (N,R,1,1) = (N,R, 1,D)x(N,R, D,1)
            k2 = k2.reshape(k2.shape[0], k2.shape[1])                   # (N,R)

    return dis_pql, cos_kr, k2, vcell, self_edges, edges, wk_pql


def compute_lattice_distances_for_cuda(
    pos, batch, trans_vec, sizes, lattice_range, cutoff_radius=0,
    output_real=False,
    output_reci=False):
    # pos: (L,D)
    # batch: (L)
    # trans_vec: (N,D,D)
    # range: int

    D = pos.shape[-1]
    factory_kywd = {'dtype':pos.dtype, 'device': pos.device}
    factory_kywd_long = {'dtype':torch.long, 'device': pos.device}

    # split flat-batched data
    if sizes is None:
        sizes = torch.zeros(trans_vec.shape[0], **factory_kywd_long)
        sizes.scatter_add_(0, batch, torch.ones_like(batch))
    if torch.is_tensor(sizes):
        sizes = sizes.tolist()
    edges = get_edge_index(sizes, pos.device)
    
    pos_p_q = dist2_min = kr_base = recip_vec = vcell = None

    vcell = det_3x3(trans_vec)
    vcell = vcell.contiguous()
    recip_vec = torch.stack([
        torch.cross(trans_vec[:, 1], trans_vec[:, 2], dim=1),
        torch.cross(trans_vec[:, 2], trans_vec[:, 0], dim=1),
        torch.cross(trans_vec[:, 0], trans_vec[:, 1], dim=1),
    ], dim=1)*(2.0*math.pi/vcell[:,None,None])
    recip_vec = recip_vec.contiguous()

    pos_p_q = pos[edges[1]] - pos[edges[0]]                         # (E,D)
    pos_p_q = pos_p_q.contiguous()

    if output_real and cutoff_radius >= 0.0:
        # Compute the nearest-neighbor distance for each atom, given lattice range.
        def shortest_distance(r):
            grids = torch.arange(-r, r+1, **factory_kywd)
            grids = torch.stack(torch.meshgrid([grids]*D, indexing='ij'), dim=-1)
            # grids: (2r+1,2r+1,2r+1,D)
            grids = grids.reshape(-1, D)
            # grids: ((2r+1)^3, D) = (R, D)

            b2e = batch[edges[0]]                                           # (B)[b2e] -> (E)
            lattice = grids @ trans_vec                                     # (B,R,D)   = (  R,D)x(B, D,D)
            pos_pql = pos_p_q[:,None] + lattice[b2e]                        # (E,R,D)   = (E, 1,D)+(E, R,D)
            del lattice, b2e
            d2min = (pos_pql*pos_pql).sum(axis=2).min(axis=1)[0]        # (E)
            d2min = d2min.contiguous()
            return d2min
        
        dist2_min = None
        # try:
        #     try:
        #         from .cuda_funcs.minimum_distance import compute_minimum_distance
        #         dist2_min = compute_minimum_distance(pos_p_q, trans_vec, batch, edges, torch.norm(recip_vec, 2, -1), cutoff_radius)
        #     except:
        #         if cutoff_radius == 0.0:
        #             dist2_min = shortest_distance(lattice_range)
        # except:
        #     dist2_min = None

    if output_reci:
        b2e = batch[edges[0]]                                       # (N)[b2e] -> (E)
        # [<r,b1>, <r,b2>, <r,b3>]
        kr_base = recip_vec[b2e] @ pos_p_q[...,None]                # (E,D,1)   = (E, D,D)x(E, D,1)
        kr_base.squeeze_(2)                                         # (E,D)
        kr_base = kr_base.contiguous()

    return pos_p_q, dist2_min, kr_base, recip_vec, vcell, edges


def compute_lattice_distances_(
    pos, batch, trans_vec, lattice_range,
    output_dis=True,
    output_rec=False):
    # pos: (samples, sdim)
    # batch: (sammples, )
    # trans_vec: (batch, sdim, sdim)
    # range: int
    sdim = pos.shape[-1]
    factory_kywd = {'dtype':pos.dtype, 'device': pos.device}
    factory_kywd_long = {'dtype':torch.long, 'device': pos.device}

    # split flat-batched data
    sizes = torch.zeros(trans_vec.shape[0], **factory_kywd_long)
    sizes.scatter_add_(0, batch, torch.ones_like(batch))
    sizes = sizes.tolist()
    pos = pos.split_with_sizes(sizes, 0)

    # define the range of lattice to be considered
    grids = torch.arange(-lattice_range, lattice_range+1, **factory_kywd)
    grids = torch.stack(torch.meshgrid([grids]*sdim), dim=-1)
    # grids: (2*R+1,2*R+1,2*R+1,sdim)
    grids = grids.reshape(-1, sdim)
    # grids: ((2*R+1)^3, sdim)

    Vcell = None
    if output_rec:
        Vcell = torch.det(trans_vec).abs()
        reciprocal_vec = torch.cat([
            torch.cross(trans_vec[:, 1:2], trans_vec[:, 2:3], dim=2),
            torch.cross(trans_vec[:, 2:3], trans_vec[:, 0:1], dim=2),
            torch.cross(trans_vec[:, 0:1], trans_vec[:, 1:2], dim=2),
        ], dim=1)*(2.0*math.pi/Vcell[:,None,None])
        half_grids = grids[grids.shape[0]//2:]
        assert abs(half_grids + grids[0:grids.shape[0]//2+1].flip(0)).sum() == 0

    rel_dis_rep = [] if output_dis else None
    rec_cos_kr = [] if output_rec else None
    rec_k2 = [] if output_rec else None
    for i,p in enumerate(pos):
        num = p.shape[0]
        rel_pos = p.reshape(1,num,sdim) - p.reshape(num,1,sdim) # (q-sam, k-sam, sdim)
        rel_pos = rel_pos.reshape(num*num, sdim)                # (sam^2, sdim)
        pos_rep = grids @ trans_vec[i]                          # ((2R+1)^3, sdim)
        pos_rep = pos_rep[None] + rel_pos[:,None]               # (sam^2, (2R+1)^3, sdim)
        dis_rep = torch.norm(pos_rep, dim=-1)                   # (sam^2, (2R+1)^3)
        if output_dis:
            rel_dis_rep.append(dis_rep)

        if output_rec:
            # sum_k pi^(3/2)/Vcell * exp(-k^2/4a^2)/a^3 * exp(i<k,r>)
            # = pi^(3/2)/Vcell (1 + 2/a^3 * sum_k' exp(-k^2/4a^2) * cos(<k,r>))
            
            # [b1, b2, b3]^T
            bMat = reciprocal_vec[i]        
            # [<r,b1>, <r,b2>, <r,b3>]
            cos_kr = rel_pos @ bMat.t()                     # (sam^2, sdim)
            
            # <k,r> = n1*<r,b1> + n2*<r,b2> + n3*<r,b3>]
            cos_kr = half_grids @ cos_kr[...,None]          # (hgrids, sdim) x (sam^2, sdim, 1)
                                                            #=(sam^2, hgrids, 1)
            cos_kr = cos_kr.squeeze(-1)                     # (sam^2, hgrids)
            cos_kr = torch.cos(cos_kr)
            
            # [<b1,b1>, <b2,b2>, <b3,b3>]
            # k = n1*b1 + n2*b2 + n3*b3
            k2 = (half_grids@bMat)                          # (hgrids, sdim) x (sdim, sdim)
                                                            #=(hgrids, sdim)
            k2 = k2[:,None,:] @ k2[:,:,None]                # (hgrids,1,sdim) x (hgrids,sdim,1)
                                                            #=(hgrids, 1, 1)
            k2 = k2.reshape(1,-1)                           # (1, hgrids)

            rec_cos_kr.append(cos_kr)
            rec_k2.append(k2)

    edges = get_edge_index(sizes, factory_kywd['device'])

    rel_dis_rep = torch.cat([x for x in rel_dis_rep], dim=0) if rel_dis_rep is not None else None
    rec_cos_kr = torch.cat([x for x in rec_cos_kr], dim=0) if rec_cos_kr is not None else None
    rec_k2 = torch.cat([x for x in rec_k2], dim=0) if rec_k2 is not None else None

    if rec_cos_kr is not None:
        rec_cos_kr[:,1:] += rec_cos_kr[:,1:]
        
    return rel_dis_rep, rec_cos_kr, rec_k2, Vcell, edges, sizes

class IndexedLatticeformerEncoder(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        t_fixup_init: if ``True``, use the initialization scheme proposed by Huang et al. 2020 in
            "Improving Transformer Optimization Through Better Initialization". Default: ``False``.
        no_layer_norm: if ``True`` all the layer norm layers in the module are removed.
            Should be used with t_fixup_init=True. Default: ``False``.
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 t_fixup_init = False, no_layer_norm = False,
                 lattice_params:LatticeformerParams=LatticeformerParams(),
                 k_dim = 0,
                 v_dim = 0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}
        super(IndexedLatticeformerEncoder, self).__init__()
        assert lattice_params.domain in ("real", "reci", "RECI", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real")
        self.lattice_params = lattice_params
        
        self.layers = ModuleList([
            IndexedLatticeformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                      activation, layer_norm_eps, norm_first,
                                      no_layer_norm,
                                      params = lattice_params.getLayerParameters(i),
                                      k_dim = k_dim,
                                      v_dim = v_dim,
                                      **factory_kwargs)
            for i in range(num_encoder_layers)
        ])
        if t_fixup_init:
            for layer in self.layers:
                layer.fixup_initialization(num_encoder_layers)

        self.num_layers = num_encoder_layers
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else None

        self.d_model = d_model
        self.nhead = nhead

        
    def forward(self, src: Tensor, pos: Tensor, batch: Tensor, trans: Tensor, sizes: Tensor, onehot: Tensor=None) -> Tensor:
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to the encoder (required).
            edges: attention edges (required).
        Shape:
            - src: :math:`(S, E)`.
            - batch: `(S)`.
            - dist: :math:`(M, R)`, where M is the number of edges.
            - edges: :math:`(2, M)`.
            - output: :math:`(T, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, src_mask=src_mask)
        """

        # 
        batch = batch - batch[0]
        dist2, cos_kr, k2, vcell, selfe, edges, pe_wave = \
            compute_lattice_distances(
                pos, batch, trans, sizes,
                self.lattice_params.lattice_range,
                output_dis=self.lattice_params.domain in ("real", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real"),
                output_rec=self.lattice_params.domain in ("reci", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real", "RECI"),
                exclude_self=self.lattice_params.exclude_self,
                dim_pe_wave=max(self.lattice_params.value_pe_wave_real, self.lattice_params.value_pe_wave_reci),
            )
        if k2 is not None and torch.isnan(k2).any():
            print("nan in k2")

        # exp(-a^2 r^2)
        if dist2 is not None:
            #dist2 /= self.scale_real
            dist2 *= dist2
        if k2 is not None:
            #k2 *= (self.scale_reci**2)
            #vcell /= (self.scale_reci**3)
            pass

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        output = src

        for mod in self.layers:
            output = mod(output, batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot)

        if self.norm is not None:
            output = self.norm(output)

        return output

class IndexedLatticeformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead:int,  dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 no_layer_norm = False,
                 params = LatticeformerParams(),
                 k_dim = 0,
                 v_dim = 0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}

        assert params.domain in ("real", "reci", "RECI", "multihead")
        self.domain = params.domain

        super(IndexedLatticeformerEncoderLayer, self).__init__()
        self.self_attn = IndexedLatticeMultiheadAttention(
            d_model, nhead, dropout=dropout,
            kdim=k_dim,
            vdim=v_dim,
            params=params,
            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout) if dropout>0 else lambda x: x
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else (lambda x: x)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else (lambda x: x)
        self.dropout1 = Dropout(dropout) if dropout>0 else lambda x: x
        self.dropout2 = Dropout(dropout) if dropout>0 else lambda x: x

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
            
        self.add_zero_attn = False #add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            constant_(self.linear2.bias, 0)
            
    def fixup_initialization(self, num_layers):
        temp_state_dic = {}
        en_layers = num_layers

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        "self_attn.v_proj_weight",
                        # "self_attn.pe_dist_proj",
                        # "self_attn.pe_wave_proj",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(IndexedLatticeformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, batch: Tensor, edges: Tensor,
        dist2: Tensor=None,
        cos_kr: Tensor=None, k2: Tensor=None, vcell: Tensor=None, selfe: Tensor=None, pe_wave: Tensor=None,
        onehot:Tensor=None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to modulate attentions (required).
            edges: the attention edges (required).
        Shape:
            see the docs in Transformer class.
        """
        if self.domain in ("real", "multihead"):
            assert dist2 is not None
        
        if self.domain in ("reci", "multihead", "RECI"):
            assert cos_kr is not None and k2 is not None and vcell is not None

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, batch: Tensor, edges: Tensor,
        dist2: Tensor=None,
        cos_kr: Tensor=None, k2: Tensor=None, vcell: Tensor=None, selfe:Tensor=None, pe_wave:Tensor=None, onehot:Tensor=None) -> Tensor:
        x = self.self_attn(x, x, x, batch, batch, edges, dist2, cos_kr, k2, vcell, selfe, pe_wave, onehot, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

from .indexed_lattice_multi_head_attention_cuda import compile_kernels

class CrystalformerEncoderCUDA(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        t_fixup_init: if ``True``, use the initialization scheme proposed by Huang et al. 2020 in
            "Improving Transformer Optimization Through Better Initialization". Default: ``False``.
        no_layer_norm: if ``True`` all the layer norm layers in the module are removed.
            Should be used with t_fixup_init=True. Default: ``False``.
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 t_fixup_init = False, no_layer_norm = False,
                 lattice_params:LatticeformerParams=LatticeformerParams(),
                 k_dim = 0,
                 v_dim = 0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}
        super(CrystalformerEncoderCUDA, self).__init__()
        assert lattice_params.domain in ("real", "reci", "RECI", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real")
        self.lattice_params = lattice_params
        
        self.layers = ModuleList([
            CrystalformerEncoderLayerCUDA(d_model, nhead, dim_feedforward, dropout,
                                      activation, layer_norm_eps, norm_first,
                                      no_layer_norm,
                                      params = lattice_params.getLayerParameters(i),
                                      k_dim = k_dim,
                                      v_dim = v_dim,
                                      **factory_kwargs)
            for i in range(num_encoder_layers)
        ])
        if t_fixup_init:
            for layer in self.layers:
                layer.fixup_initialization(num_encoder_layers)

        self.num_layers = num_encoder_layers
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else None

        self.d_model = d_model
        self.nhead = nhead
        v_head_dim = v_dim if v_dim>0 else d_model//nhead
        k_head_dim = k_dim if k_dim>0 else d_model//nhead
        compile_kernels(
            lattice_params.lattice_range, nhead, 
            k_head_dim,
            lattice_params.value_pe_dist_real, v_head_dim,
            lattice_params.minimum_range)


        
    def forward(self, src: Tensor, pos: Tensor, batch: Tensor, trans: Tensor, sizes: Tensor, onehot: Tensor=None) -> Tensor:
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to the encoder (required).
            edges: attention edges (required).
        Shape:
            - src: :math:`(S, E)`.
            - batch: `(S)`.
            - dist: :math:`(M, R)`, where M is the number of edges.
            - edges: :math:`(2, M)`.
            - output: :math:`(T, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, src_mask=src_mask)
        """

        # for multi-gpu execution, adjust batch indices
        batch = batch - batch[0]
        P = self.lattice_params
        pos_ij, dist2_min, kr_base, rvecs, vcell, edges = compute_lattice_distances_for_cuda(
            pos, batch, trans, sizes, P.lattice_range,
            -1, #P.scale_real*P.gauss_lb_real**(-0.5)*P.adaptive_cutoff_sigma,
            output_real=self.lattice_params.domain in ("real", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real"),
            output_reci=self.lattice_params.domain in ("reci", "multihead", "real-reci", "reci-real", "real-RECI", "RECI-real", "RECI"),
        )

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        output = src

        for mod in self.layers:
            output = mod(output, batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot)

        if self.norm is not None:
            output = self.norm(output)

        return output

class CrystalformerEncoderLayerCUDA(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead:int,  dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 no_layer_norm = False,
                 params = LatticeformerParams(),
                 k_dim = 0,
                 v_dim = 0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        factory_kwargs = {}

        assert params.domain in ("real", "reci", "RECI", "multihead")
        self.domain = params.domain

        super(CrystalformerEncoderLayerCUDA, self).__init__()
        self.self_attn = CrystalformerMultiheadAttentionCUDA(
            d_model, nhead, dropout=dropout,
            kdim=k_dim,
            vdim=v_dim,
            params=params,
            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout) if dropout>0 else Identity()
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else Identity()
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) if not no_layer_norm else Identity()
        self.dropout1 = Dropout(dropout) if dropout>0 else Identity()
        self.dropout2 = Dropout(dropout) if dropout>0 else Identity()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
            
        self.add_zero_attn = False #add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            constant_(self.linear2.bias, 0)
            
    def fixup_initialization(self, num_layers):
        temp_state_dic = {}
        en_layers = num_layers

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        "self_attn.v_proj_weight",
                        # "self_attn.pe_dist_proj",
                        # "self_attn.pe_wave_proj",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CrystalformerEncoderLayerCUDA, self).__setstate__(state)

    def forward(self, src: Tensor, batch: Tensor, edges: Tensor,
        pos_ij: Tensor=None, dist2_min:Tensor=None, trans: Tensor=None,
        kr_base: Tensor=None, rvecs: Tensor=None, vcell: Tensor=None,
        onehot:Tensor=None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            batch: batch indices of the sequence (required).
            dist: the distances of lattice points to modulate attentions (required).
            edges: the attention edges (required).
        Shape:
            see the docs in Transformer class.
        """
        if self.domain in ("real", "multihead"):
            assert pos_ij is not None and trans is not None
        
        if self.domain in ("reci", "multihead", "RECI"):
            assert kr_base is not None and rvecs is not None and vcell is not None

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, batch: Tensor, edges: Tensor,
        pos_ij: Tensor=None, dist2_min:Tensor=None, trans: Tensor=None,
        kr_base: Tensor=None, rvecs: Tensor=None, vcell: Tensor=None,
        onehot:Tensor=None) -> Tensor:
        x = self.self_attn(x, x, x, batch, batch, edges, pos_ij, dist2_min, trans, kr_base, rvecs, vcell, onehot, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


