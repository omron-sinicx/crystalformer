
from operator import truediv
import warnings
from typing import List, Optional, Tuple

import math
import torch
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_, normal_
from torch.nn import Parameter, Module
import torch.nn.functional as F
from .latticeformer_params import LatticeformerParams

# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        try:
            super().__init__(in_features, out_features, bias=bias,
                            device=device, dtype=dtype)
        except:
            super().__init__(in_features, out_features, bias=bias)


#
# multihead attention
#

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    Dq, Dk, Dv = w_q.size(0), w_k.size(0), w_v.size(0)
    assert w_q.shape[1] == Eq, f"expecting query weights shape of (*, {Eq}), but got {w_q.shape}"
    assert w_k.shape[1] == Ek, f"expecting key weights shape of (*, {Ek}), but got {w_k.shape}"
    assert w_v.shape[1] == Ev, f"expecting value weights shape of (*, {Ev}), but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Dq,), f"expecting query bias shape of {(Dq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Dk,), f"expecting key bias shape of {(Dk,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Dv,), f"expecting value bias shape of {(Dv,)}, but got {b_v.shape}"
    
    # F.linear(x, W, b) = xW^T + b
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batch: Tensor,
    batch_kv: Tensor,
    edges: Tensor,
    attn_weights: Optional[Tensor] = None,
    values: Tensor = None,
    dropout_p: float = 0.0,
    norm_func_mode: int = 0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        edges: index pairs (i,j) to define attentions between q and p,v.
        attn_weights: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(Nt, B, E)` where Nt is the target sequence length, B is batch size,
            and E is embedding dimension.
        - key: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - value: :math:`(Ns, B, E)` where Ns is the source sequence length, B is batch size,
            and E is embedding dimension.
        - edges: :math:`(2, M)` where M is the edge num.
        - attn_weights: `(M, B)` where M in the edge num, B is batch size.
        - Output: attention values have shape :math:`(Nt, B, E)`; attention weights
            have shape :math:`(M, B)` where M in the edge num, B is batch size.
    """
    Nt, B, E = q.shape
    q = q / math.sqrt(E)
    # (M, B, E) x (M, B, E) -> (M, B)
    attn = (q[edges[0]]*k[edges[1]]).sum(dim=-1)

    #flag = torch.are_deterministic_algorithms_enabled()         
    #torch.use_deterministic_algorithms(False)
    bsz = batch.max().item()+1
    q_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
    q_sizes.scatter_add_(0, batch, torch.ones_like(batch))

    if batch_kv is batch:
        k_sizes = q_sizes
    else:
        k_sizes = torch.zeros(bsz, dtype=torch.long, device=q.device)
        k_sizes.scatter_add_(0, batch_kv, torch.ones_like(batch_kv))
    # This is because self-attention has the same number of queries and keys (sys_size).
    edg_sizes = q_sizes*k_sizes

    q_sizes = q_sizes.tolist()
    k_sizes = k_sizes.tolist()
    edg_sizes = edg_sizes.tolist()
    #torch.use_deterministic_algorithms(flag)

    if True:
        # The scaled_dot operation involves the summations along the key axis
        # whose size varies among batch samples. So we split concatenated data 
        # into a list of batch samples and apply the scaled_dot for each sample.
        # We could do the same without the splitting & looping by using scatter_add,
        # but we rather avoid scatter_add as it breaks reproducibility in backprop.
        if attn_weights is None:
            attn_weights = 0

        abs_mode = abs(norm_func_mode)
        if abs_mode == 0 or norm_func_mode == "default":
            # standard normalization for all system points.
            attn += attn_weights
            attn = torch.split_with_sizes(attn, edg_sizes)
            attn = torch.cat([F.softmax(a.view(qs,ks,-1),dim=1).view(qs*ks,-1) for a,qs,ks in zip(attn,q_sizes,k_sizes)])
        elif abs_mode == 1 or norm_func_mode == "softmax":
            # partial normalization for only primitive points.
            aw = torch.split_with_sizes(torch.exp(attn_weights), edg_sizes)
            attn = torch.split_with_sizes(attn, edg_sizes)
            attn = torch.cat([F.softmax(a.view(qs,ks,-1),dim=1).view(qs*ks,-1)*w for a,w,qs,ks in zip(attn,aw,q_sizes,k_sizes)])
        elif abs_mode == 2 or norm_func_mode == "exp":
            # no normalization
            # NOTE: this mode causes nan.
            attn += attn_weights
            attn.exp_()
        elif abs_mode == 3 or norm_func_mode == "linear":
            # no nomalization with linear attention
            attn = attn * torch.exp(attn_weights)
        elif abs_mode == 4 or norm_func_mode == "sigmoid":
            # no nomalization with sigmoid attention
            attn = torch.sigmoid(attn) * torch.exp(attn_weights)
        elif abs_mode == 5 or norm_func_mode == "softplus":
            # no nomalization with softplus attention
            attn = torch.nn.functional.softplus(attn) * torch.exp(attn_weights)
        elif abs_mode == 6 or norm_func_mode == "elu":
            # no nomalization with elu attention
            attn = torch.nn.functional.elu(attn) * torch.exp(attn_weights)
        elif abs_mode == 7 or norm_func_mode == "elu+1":
            # no nomalization with elu attention
            attn = (torch.nn.functional.elu(attn)+1) * torch.exp(attn_weights)
        elif abs_mode == 8 or norm_func_mode == "abs":
            # no nomalization with elu attention
            attn = abs(attn) * torch.exp(attn_weights)
        else:
            raise ValueError(f"norm_func_mode must be 0 to 4 but {norm_func_mode} is given.")

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        # (M, B, 1) x (M, B, E) -> (q_len, B, E)
        if values is None:
            output = attn[...,None]*v[edges[1]]
        else:
            output = attn[...,None]*(v[edges[1]] + values)
        output = torch.split_with_sizes(output, edg_sizes)
        output = torch.cat([o.view((qs,ks)+o.shape[1:]).sum(dim=1) for o,qs,ks in zip(output,q_sizes,k_sizes)])
    else:
        if attn_weights is not None:
            attn += attn_weights
    
        # This code was slower (3.65 it/sec vs 3.95 it/sec).
        attn = torch.split_with_sizes(attn, edg_sizes)
        v = torch.split_with_sizes(v, sys_sizes)
        output = []
        for a,v,s in zip(attn,v,sys_sizes):
            a = F.softmax(a.view(s,s,-1), dim=1)
            if dropout_p > 0.0:
                a = F.dropout(a, p=dropout_p)
            # (Nt,Nt,B)x(1,Nt,B,E).sum(dim=1) -> # (Nt,B,E)
            output.append((a[...,None]*v[None]).sum(dim=1))
        output = torch.cat(output)

    sizes = (q_sizes, k_sizes, edg_sizes)
    return output, attn, sizes


def _mha_shape_check(
    query: Tensor, key: Tensor, value: Tensor, edges: Tensor,
    dist2: Tensor, cos_kr: Tensor, k2: Tensor, vcell: Tensor, selfe: Tensor, num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `dist`, and `edges`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    assert query.dim() == 2, f"Expected `query` to be 2-D but got a {query.dim()}-D tensor."
    assert key.dim() == 2, f"Expected `key` to be 2-D but got a {key.dim()}-D tensor."
    assert value.dim() == 2, f"Expected `value` to be 2-D but got a {value.dim()}-D tensor."
    
    assert edges.dim() == 2, f"Expected `edges` to be 2-D but got a {edges.dim()}-D tensor."
    assert edges.shape[0] == 2
    assert edges.dtype == torch.long
    
    if dist2 is not None:
        assert dist2.dim() == 2, f"Expected `dist` to be 2-D but got a {dist2.dim()}-D tensor."
        assert edges.shape[1] == dist2.shape[0]

    if cos_kr is not None:
        assert cos_kr.dim() == 2, f"Expected `dist2` to be 2-D but got a {cos_kr.dim()}-D tensor."
        assert k2.dim() == 2, f"Expected `k2` to be 2-D but got a {k2.dim()}-D tensor."
        assert vcell.dim() == 1, f"Expected `vcell` to be 1-D but got a {vcell.dim()}-D tensor."
        assert cos_kr.shape[0] == edges.shape[1]
        assert k2.shape[0] == vcell.shape[0]
        assert k2.shape[1] == cos_kr.shape[1]
    if selfe is not None:
        assert selfe.dim() == 1, f"Expected `selfe` to be 1-D but got a {selfe.dim()}-D tensor."

def logsumexp_xy(x, y, dim):
    return torch.logsumexp(x*y, dim=dim)


class LogSumExpXYFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, dim, keepdim=False):
        # x: (edges,     1, mirrors)
        # y: (edges, heads,       1)
        # z = log(sum_i exp(xi*yi))
        
        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        z = torch.logsumexp(x*y, dim=dim, keepdim=keepdim)

        ctx.save_for_backward(x, y, z)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return z

    @staticmethod
    def backward(ctx, gz):
        x, y, z = ctx.saved_tensors
        gx = gy = None

        # z = log(sum_i exp(xi*yi))
        # dz/dxj = yj*exp(xj*yj) / (sum_i exp(xi*yi))
        #        = yj*exp(xj*yj) / exp(z)
        #        = yj*exp(xj*yj - z)
        if not ctx.keepdim:
            z = z.unsqueeze(ctx.dim)
            gz = gz.unsqueeze(ctx.dim)
        g = gz*torch.exp(x*y-z)

        if ctx.needs_input_grad[0]:
            gx = y*g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        if ctx.needs_input_grad[1]:
            gy = x*g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return gx, gy, None, None

class AverageBySoftmaxXYFunc_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, x, y, dim, W=None, keepdim=False):
        # a: (edges,     1, mirrors, K)
        # x: (edges,     1, mirrors, 1)
        # y: (edges, heads,       1, 1)
        # W: (heads, head_dim, K)
        # z = log(sum_i exp(xi*yi))
        
        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        p = torch.softmax(x*y, dim=dim)
        # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
        # p: (edges, heads, mirrors, 1) - > (edges, heads, 1, mirrors)
        # a: (edges,     1, mirrors, K)
        E, H, R, _ = p.shape
        p = p.reshape(E, H, 1, R)
        # Below is equivalent to: z = p @ a
        # but unloop it along the heads axis to reduce the peak memory usage.
        # z: [edges, 1, 1, K] x heads
        z = [p[:, i:i+1] @ a for i in range(H)]
        if W is not None:
            assert W.dim() in (2,3,4)
            W_ = W
            if W_.dim()==2:
                W_ = W_.unsqueeze(0)
            if W_.dim()==3:
                W_ = W_.unsqueeze(0)
            if W_.shape[1]==1:
                W_ = W_.expand(W_.shape[0], H, W_.shape[-2], W_.shape[-1])
            W_ = W_.transpose(2, 3)
            
            z0 = torch.cat(z, dim=1)
            # [(edges, 1, 1, K) x (K, D)] = [(edges, 1, 1, D)]
            z = [z[i] @ W_[:, i:i+1] for i in range(H)]
            del W_

        z = torch.cat(z, dim=1)
        if not keepdim:
            z.squeeze_(2)

        if W is None:
            ctx.save_for_backward(a, x, y)
        else:
            ctx.save_for_backward(a, x, y, W, z0)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return z

    @staticmethod
    def backward(ctx, gz):
        if len(ctx.saved_tensors) == 3:
            a, x, y = ctx.saved_tensors
            W = z0 = None
        else:
            a, x, y, W, z0 = ctx.saved_tensors
        ga = gx = gy = gW = None

        # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
        #    = a*p where p = softmax(x*y).
        # dz/dxi = p*y*(a*g - (p*a*g).sum())
        if not ctx.keepdim:
            gz = gz.unsqueeze(ctx.dim)

        if W is not None:
            # gz: (E, H, 1, D) -> (E,H,K,D) -> (H,K,D)
            # W : (H, D, K)
            # z0: (E, H, 1, K)
            #gW = [(gz_*z0_).sum(0) for gz_, z0_ in zip(torch.split(gz.transpose(2,3), 1, 1), torch.split(z0, 1, 1))]
            #gW = torch.cat(gW, 0)
            # (H,1,D,E) x (H,1,E,K) -> (H,1,D,K)
            gW = gz.permute(1,2,3,0) @ z0.permute(1,2,0,3)
            gW.squeeze_(1)
            #gW = (gz.transpose(2,3)*z0).sum(0)

            # gz: (E,H,1,D)x(H,D,K) -> (E,H,1,K)
            gz0 = gz @ W    # Compute it later as this step is costly.
            #gz0 = None
        else:
            gz0 = gz

        # p: (E,H,R,1)
        p = torch.softmax(x*y, dim=ctx.dim)

        if ctx.needs_input_grad[0]:
            # ga: (E,1,R,K)
            ga = p*gz0 if gz0 is not None else p*(gz @ W)
            dims = [i for i,(si,so) in enumerate(zip(ga.shape,a.shape)) if si!=so]
            if len(dims) > 0:
                ga = ga.sum(dim=dims, keepdim=True)

        if False:
            # This code requires huge temporary memory
            # for computing g*a of size (edges, heads, mirrors, K).
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)
            g = a*gz0
            g -= (g*p).sum(dim=ctx.dim, keepdim=True)
            g *= p

        else:
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)

            # Unloop the last axis (size K)
            K = a.shape[-1]
            s = torch.broadcast_shapes(a.shape[:3], gz.shape[:3])
            g = torch.zeros(s, device=gz.device, dtype=gz.dtype)
            if gz0 is not None:
                for i in range(K):
                    g += a[...,i]*gz0[...,i]
            else:
                # Because gz has not been updated as gz@W, compute it here in the loop.
                # gz@W : (E,H,1,D)x(H,D,K)
                for i in range(K):
                    g += a[...,i]*(gz@W[...,i:i+1]).squeeze(-1)
            g.unsqueeze_(-1)
            g -= (g*p).sum(dim=ctx.dim, keepdim=True)
            g *= p
            del p, gz, gz0
            
        # gx = y*p*(a*gz - (a*gz*p).sum())
        if ctx.needs_input_grad[1]:
            gx = y*g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        # gy = x*p*(a*gz - (a*gz*p).sum())
        # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
        if ctx.needs_input_grad[2]:
            gy = x*g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return ga, gx, gy, None, gW, None

class AverageBySoftmaxXYFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, x, y, dim, keepdim=False):
        # a: (edges,     1, mirrors, K)
        # x: (edges,     1, mirrors, 1)
        # y: (edges, heads,       1, 1)
        # W: (heads, head_dim, K)
        # z = log(sum_i exp(xi*yi))
        
        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        p = torch.softmax(x*y, dim=dim)
        # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
        # p: (edges, heads, mirrors, 1) - > (edges, heads, 1, mirrors)
        # a: (edges,     1, mirrors, K)
        p = p.view(p.shape[0], p.shape[1], 1, p.shape[2])
        # Below is equivalent to: z = p @ a
        # but unloop it along the heads axis to reduce the peak memory usage.
        # z: [edges, 1, 1, K] x heads
        z = [p[:, i:i+1] @ a for i in range(p.shape[1])]
        z = torch.cat(z, dim=1)
        if not keepdim:
            z.squeeze_(2)

        ctx.save_for_backward(a, x, y)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return z

    @staticmethod
    def backward(ctx, gz):
        a, x, y = ctx.saved_tensors
        ga = gx = gy = None

        p = torch.softmax(x*y, dim=ctx.dim)

        # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
        #    = a*p where p = softmax(x*y).
        # dz/dxi = p*y*(a*g - (p*a*g).sum())
        if not ctx.keepdim:
            gz = gz.unsqueeze(ctx.dim)

        if ctx.needs_input_grad[0]:
            ga = p*gz
            dims = [i for i,(si,so) in enumerate(zip(ga.shape,a.shape)) if si!=so]
            if len(dims) > 0:
                ga = ga.sum(dim=dims, keepdim=True)

        if False:
            # This code requires huge temporary memory
            # for computing g*a of size (edges, heads, mirrors, K).
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)
            g = a*gz
            g -= (g*p).sum(dim=ctx.dim, keepdim=True)
            g *= p

            if ctx.needs_input_grad[1]:
                gx = y*g
                # if x was broadcast in forward, sum the gradients along the broadcast axes.
                dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
                if len(dims) > 0:
                    gx = gx.sum(dim=dims, keepdim=True)

            if ctx.needs_input_grad[2]:
                gy = x*g
                # if y was broadcast in forward, sum the gradients along the broadcast axes.
                dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
                if len(dims) > 0:
                    gy = gy.sum(dim=dims, keepdim=True)
            
        else:
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)

            # Unloop the last axis (size K)
            K = a.shape[-1]
            s = torch.broadcast_shapes(a.shape, gz.shape)
            g = torch.zeros(s[0:3], device=gz.device, dtype=gz.dtype)
            for i in range(K):
                g += a[...,i]*gz[...,i]
            g.unsqueeze_(-1)
            g -= (g*p).sum(dim=ctx.dim, keepdim=True)
            g *= p
            
            # gx = y*p*(a*gz - (a*gz*p).sum())
            if ctx.needs_input_grad[1]:
                gx = y*g
                # if x was broadcast in forward, sum the gradients along the broadcast axes.
                dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
                if len(dims) > 0:
                    gx = gx.sum(dim=dims, keepdim=True)

            # gy = x*p*(a*gz - (a*gz*p).sum())
            # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
            if ctx.needs_input_grad[2]:
                gy = x*g
                # if y was broadcast in forward, sum the gradients along the broadcast axes.
                dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
                if len(dims) > 0:
                    gy = gy.sum(dim=dims, keepdim=True)

        return ga, gx, gy, None, None

def _get_basis_func(m, K, x, method, width_scale=1.0):
    if method is None or method in ("", "cgcnn"):
        a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        a = a - x**0.5
        a /= ((2**0.5)*(width_scale*m/K))
        a *= a
        torch.neg_(a)
        torch.exp_(a)
        return a
    if method == "physnet":
        # vij_k = exp (−βk (exp(−rij ) − µk)**2)
        # βk: a fixed βk = [2 (1 − exp(−rcut))/K ]**−2
        # µk: equally spaced between exp(−rcut) and 1 with rcut = 10
        a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        w = (1.0-math.exp(-m))/K
        beta = (2.0*w)**-2
        






class AverageGaussiansBySoftmaxXYFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, K, x, y, dim, keepdim=False, basis_func:str="", width_scale=1.0):
        # m: float
        # K: int
        # x: (edges,     1, mirrors, 1)
        # y: (edges, heads,       1, 1)
        # W: (heads, head_dim, K)
        # z = log(sum_i exp(xi*yi))
        
        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        p = torch.softmax(x*y, dim=dim)
        # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
        # p: (edges, heads, mirrors, 1) - > (edges, heads, 1, mirrors)
        # a: (edges,     1, mirrors, K)
        
        # a = exp((mu_k - dist)^2/(2*mu_0^2))
        a = _get_basis_func(m, K, x, basis_func, width_scale)
        # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        # a = a - x**0.5
        # a /= ((2**0.5)*m/K)
        # a *= a
        # torch.neg_(a)
        # torch.exp_(a)

        p = p.view(p.shape[0], p.shape[1], 1, p.shape[2])
        # Below is equivalent to: z = p @ a
        # but unloop it along the heads axis to reduce the peak memory usage.
        # z: [edges, 1, 1, K] x heads
        z = [p[:, i:i+1] @ a for i in range(p.shape[1])]
        z = torch.cat(z, dim=1)
        if not keepdim:
            z.squeeze_(2)

        ctx.save_for_backward(x, y)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.m = m
        ctx.K = K
        ctx.basis_func = basis_func
        ctx.width_scale = width_scale
        return z

    @staticmethod
    def backward(ctx, gz):
        x, y = ctx.saved_tensors
        gx = gy = None
        K = ctx.K
        m = ctx.m
        basis_func = ctx.basis_func
        width_scale = ctx.width_scale

        p = torch.softmax(x*y, dim=ctx.dim)

        # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
        #    = a*p where p = softmax(x*y).
        # dz/dxi = p*y*(a*g - (p*a*g).sum())
        if not ctx.keepdim:
            gz = gz.unsqueeze(ctx.dim)

        if True:
            a = _get_basis_func(m, K, x, basis_func, width_scale)
            # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
            # a = a - x**0.5
            # a /= ((2**0.5)*m/K)
            # a *= a
            # torch.neg_(a)
            # torch.exp_(a)

            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)

            # Unloop the last axis (size K) of a
            s = torch.broadcast_shapes(a.shape, gz.shape)
            g = torch.zeros(s[0:3], device=gz.device, dtype=gz.dtype)
            for i in range(K):
                g += a[...,i]*gz[...,i]
            g.unsqueeze_(-1)
            g -= (g*p).sum(dim=ctx.dim, keepdim=True)
            g *= p
            
            # gx = y*p*(a*gz - (a*gz*p).sum())
            if ctx.needs_input_grad[2]:
                gx = y*g
                # if x was broadcast in forward, sum the gradients along the broadcast axes.
                dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
                if len(dims) > 0:
                    gx = gx.sum(dim=dims, keepdim=True)

            # gy = x*p*(a*gz - (a*gz*p).sum())
            # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
            if ctx.needs_input_grad[3]:
                gy = x*g
                # if y was broadcast in forward, sum the gradients along the broadcast axes.
                dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
                if len(dims) > 0:
                    gy = gy.sum(dim=dims, keepdim=True)

        return None, None, gx, gy, None, None, None, None

class AverageGaussiansBySoftmaxXYFunc_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, K, x, y, dim, W=None, width_scale=1.0, basis_func:str="", keepdim=False):
        # a: (edges,     1, mirrors, K)
        # x: (edges,     1, mirrors, 1)
        # y: (edges, heads,       1, 1)
        # W: (edges, heads, head_dim, K)
        # z = log(sum_i exp(xi*yi))
        
        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.
        p = torch.softmax(x*y, dim=dim)
        # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
        # p: (edges, heads, mirrors, 1) - > (edges, heads, 1, mirrors)
        # a: (edges,     1, mirrors, K)
        E, H, R, _ = p.shape
        p = p.reshape(E, H, 1, R)

        # a = exp((mu_k - dist)^2/(2*mu_0^2))
        a = _get_basis_func(m, K, x, basis_func, width_scale)
        # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
        # a = a - x**0.5
        # a /= ((2**0.5)*(m/K))
        # a *= a
        # torch.neg_(a)
        # torch.exp_(a)

        # Below is equivalent to: z = p @ a
        # but unloop it along the heads axis to reduce the peak memory usage.
        # z: [edges, 1, 1, K] x heads
        z = [p[:, i:i+1] @ a for i in range(H)]
        # try:
        #     z = p @ a
        # except:
        #     z = [p[:, i:i+1] @ a for i in range(H)]
        del a

        if W is not None:
            assert W.dim() in (2,3,4)
            W_ = W
            if W_.dim()==2:
                W_ = W_.unsqueeze(0)
            if W_.dim()==3:
                W_ = W_.unsqueeze(0)
            if W_.shape[0]==1:
                W_ = W_.expand(W_.shape[0], H, W_.shape[-2], W_.shape[-1])
            W_ = W_.transpose(-2, -1)
            
            z0 = torch.cat(z, dim=1)
            z = [z[i] @ W_[:, i:i+1] for i in range(H)]
            # try:
            #     z = z0 @ W_
            # except:
            #     # [(edges, 1, 1, K) x (K, D)] = [(edges, 1, 1, D)]
            #     z = [z0[:, i:i+1] @ W_[:, i:i+1] for i in range(H)]
            del W_
        
        z = torch.cat(z, dim=1)
        
        if not keepdim:
            z.squeeze_(2)

        if W is None:
            ctx.save_for_backward(x, y)
        else:
            ctx.save_for_backward(x, y, W, z0)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.K = K
        ctx.m = m
        ctx.basis_func = basis_func
        ctx.width_scale = width_scale
        return z

    @staticmethod
    def backward(ctx, gz):
        if len(ctx.saved_tensors) == 2:
            x, y = ctx.saved_tensors
            W = z0 = None
        else:
            x, y, W, z0 = ctx.saved_tensors
        gx = gy = gW = None
        m = ctx.m
        K = ctx.K
        basis_func = ctx.basis_func
        width_scale = ctx.width_scale

        # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
        #    = a*p where p = softmax(x*y).
        # dz/dxi = p*y*(a*g - (p*a*g).sum())
        if not ctx.keepdim:
            gz = gz.unsqueeze(ctx.dim)

        if W is not None:
            # gz: (E, H, 1, D) -> (E,H,K,D) -> (H,K,D)
            # W : (E, H, D, K)
            # z0: (E, H, 1, K)
            #gW = [(gz_*z0_).sum(0) for gz_, z0_ in zip(torch.split(gz.transpose(2,3), 1, 1), torch.split(z0, 1, 1))]
            #gW = torch.cat(gW, 0)
            # (H,1,D,E) x (H,1,E,K) -> (H,1,D,K)
            if W.shape[0] == 1:
                gW = gz.permute(1,2,3,0) @ z0.permute(1,2,0,3)
                gW.transpose_(0, 1)
            else:
                gW = gz.transpose(-1, -2)*z0

            #gW = (gz.transpose(2,3)*z0).sum(0)

            # gz: (E,H,1,D)x(E,H,D,K) -> (E,H,1,K)
            try:
                gz0 = gz @ W
            except:
                # Compute later if fails due to memory shortage.
                gz0 = None
        else:
            gz0 = gz

        # p: (E,H,R,1)
        p = torch.softmax(x*y, dim=ctx.dim)

        if True:
            a = _get_basis_func(m, K, x, basis_func, width_scale)
            # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
            # a = a - x**0.5
            # a /= ((2**0.5)*m/K)
            # a *= a
            # torch.neg_(a)
            # torch.exp_(a)
        
            # a: (edges,     1, mirrors, K)
            # x: (edges,     1, mirrors, 1)
            # y: (edges, heads,       1, 1)
            # z: (edges, heads,       1, K)

            # Unloop the last axis (size K) of a
            s = torch.broadcast_shapes(a.shape[:3], gz.shape[:3])
            g = torch.zeros(s, device=gz.device, dtype=gz.dtype)
            if gz0 is not None:
                for i in range(K):
                    g += a[...,i]*gz0[...,i]
            else:
                # Because gz has not been updated as gz@W, compute it here in the loop.
                # gz@W : (E,H,1,D)x(H,D,K)
                for i in range(K):
                    g += a[...,i]*(gz@W[...,i:i+1]).squeeze(-1)
            del a
            g.unsqueeze_(-1)
            g -= (g*p).sum(dim=ctx.dim, keepdim=True)
            g *= p
            del p, gz, gz0
            
        # gx = y*p*(a*gz - (a*gz*p).sum())
        if ctx.needs_input_grad[2]:
            gx = y*g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        # gy = x*p*(a*gz - (a*gz*p).sum())
        # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
        if ctx.needs_input_grad[3]:
            gy = x*g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return None, None, gx, gy, None, gW, None, None, None

def compute_real_domain_attn(alpha, dist2, edges0):
    # alpha: (nodes, heads)
    # dist2: (edge_num, R)
    # pe_wave: (edge_num, R, K)
    # where R = (1+2r)^3 and K is the PE dim.
    attn_weights = LogSumExpXYFunc.apply(
        dist2.unsqueeze(1),
        alpha.unsqueeze(2)[edges0],
        -1
    )
    return attn_weights

def compute_reci_domain_attn(alpha, cos_kr, k2, vcell, edges0, batch):
    # We compute: sum k'[exp(-k^2/4a^2) * cos(kr)] for the real-domain func exp(-a^2*r^2).
    # Here, 1/a^2 = alpha in our code. So, 
    #   sum k'[exp(-alpha^2*k^2/4) * cos(kr)]
    # cos_kr: (E, R)
    # k2    : (N, R)
    # alpha : (L, H)
    # vcell : (N)

    #self_coef = vcell.unsqueeze(1)[batch] / (2.0*math.pi*alpha)**(3/2)   # (L, H)
    alp = alpha.unsqueeze(2)                 # (L, H, 1)
    k2 = k2.unsqueeze(1)[batch]                         # (L, 1, R)
    cos_kr = cos_kr.unsqueeze(1)                        # (E, 1, R)
    
    # NOTE: max alpha_k2 = 0 so exp(alpha_k2) does not overflow.
    alpha_k2 = alp*k2
    attn_weights = (torch.exp(alpha_k2)[edges0]*cos_kr).sum(dim=-1)
    attn_weights = torch.log(attn_weights.clamp_(1e-6))
    
    # NOTE: the following correction is required when using non-softmax attention.
    # In the paper, the correction term is a factor: (1/V)(2*pi/gamma)^(3/2).
    # Here, alpha = -1/(2*gamma). So, (2*pi/gamma)^3/2 = (-4*pi*alpha)^3/2.
    log_ci = (3/2)*(torch.log((-4*math.pi)*alpha)) - torch.log(vcell)[batch, None]
    attn_weights += log_ci[edges0]

    return attn_weights

class PeriodicPositionEncodingFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, y, dim, m, K, W=None, width_scale=1.0, basis_func:str="", keepdim=False):
        # a: (edges,     1, mirrors, K)
        # x: (edges,     1, mirrors)
        # y: (edges, heads,       1)
        # W: (edges, heads, head_dim, K)
        # Output:
        # z1 = log(sum_i exp(xi*yi))
        # z2 = log(sum_i exp(xi*yi))

        # `x*y` invokes breadcasting that results in a huge tensor.
        # This custom function does not hold x*y for backward,
        # but just hold the inputs x and y as is, which does not increase memory usage.
        # TODO: write a custom cuda kernel to avoid temporary memory allocation for x*y.  
        dim = x.dim()+dim if dim< 0 else dim      
        z1 = torch.logsumexp(x*y, dim=dim, keepdim=keepdim)

        if K > 0:
            p = torch.softmax(x*y, dim=dim)
            # z = [(p[:, i:i+1]*a).sum(dim=dim, keepdim=keepdim) for i in range(p.shape[1])]
            # p: (edges, heads, mirrors) - > (edges, heads, 1, mirrors)
            # a: (edges,     1, mirrors, K)
            E, H, R = p.shape
            p = p.reshape(E, H, 1, R)

            # a = exp((mu_k - dist)^2/(2*mu_0^2))
            a = _get_basis_func(m, K, x[...,None], basis_func, width_scale)
            # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
            # a = a - x**0.5
            # a /= ((2**0.5)*(m/K))
            # a *= a
            # torch.neg_(a)
            # torch.exp_(a)

            # Below is equivalent to: z = p @ a
            # but unloop it along the heads axis to reduce the peak memory usage.
            # z: [edges, 1, 1, K] x heads
            z = [p[:, i:i+1] @ a for i in range(H)]
            # try:
            #     z = p @ a
            # except:
            #     z = [p[:, i:i+1] @ a for i in range(H)]
            del a

            if W is not None:
                assert W.dim() in (2,3,4)
                W_ = W
                if W_.dim()==2:
                    W_ = W_.unsqueeze(0)
                if W_.dim()==3:
                    W_ = W_.unsqueeze(0)
                if W_.shape[0]==1:
                    W_ = W_.expand(W_.shape[0], H, W_.shape[-2], W_.shape[-1])
                W_ = W_.transpose(-2, -1)
                
                z0 = torch.cat(z, dim=1)
                z = [z[i] @ W_[:, i:i+1] for i in range(H)]
                # try:
                #     z = z0 @ W_
                # except:
                #     # [(edges, 1, 1, K) x (K, D)] = [(edges, 1, 1, D)]
                #     z = [z0[:, i:i+1] @ W_[:, i:i+1] for i in range(H)]
                del W_
            
            z = torch.cat(z, dim=1)
        
            if not keepdim:
                z.squeeze_(dim)
        else:
            z = None
        
        if W is None:
            ctx.save_for_backward(x, y, z1)
        else:
            ctx.save_for_backward(x, y, z1, W, z0)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.m = m
        ctx.K = K
        ctx.basis_func = basis_func
        ctx.width_scale = width_scale
        return z1, z

    @staticmethod
    def backward(ctx, gz1, gz2):
        if len(ctx.saved_tensors) == 3:
            x, y, z1 = ctx.saved_tensors
            W = z0 = None
        else:
            x, y, z1, W, z0 = ctx.saved_tensors
        gx = gy = gW = None
        m = ctx.m
        K = ctx.K
        basis_func = ctx.basis_func
        width_scale = ctx.width_scale

        # for z1
        # z = log(sum_i exp(xi*yi))
        # dz/dxj = yj*exp(xj*yj) / (sum_i exp(xi*yi))
        #        = yj*exp(xj*yj) / exp(z)
        #        = yj*exp(xj*yj - z)
        if not ctx.keepdim:
            z1 = z1.unsqueeze(ctx.dim)
            gz1 = gz1.unsqueeze(ctx.dim)
        #g = gz1*torch.exp(x*y-z1)
        p = torch.exp(x*y-z1) #torch.softmax(x*y, dim=ctx.dim)
        g = gz1*p

        if K > 0:
            # for z2
            # zi = exp(xi*yi)*ai/(sum_k exp(xk*yk))
            #    = a*p where p = softmax(x*y).
            # dz/dxi = p*y*(a*g - (p*a*g).sum())
            if not ctx.keepdim:
                gz2 = gz2.unsqueeze(ctx.dim)
            gz = gz2

            if W is not None:
                # gz: (E, H, 1, D) -> (E,H,K,D) -> (H,K,D)
                # W : (E, H, D, K)
                # z0: (E, H, 1, K)
                #gW = [(gz_*z0_).sum(0) for gz_, z0_ in zip(torch.split(gz.transpose(2,3), 1, 1), torch.split(z0, 1, 1))]
                #gW = torch.cat(gW, 0)
                # (H,1,D,E) x (H,1,E,K) -> (H,1,D,K)
                if W.shape[0] == 1:
                    gW = gz.permute(1,2,3,0) @ z0.permute(1,2,0,3)
                    gW.transpose_(0, 1)
                else:
                    gW = gz.transpose(-1, -2)*z0

                #gW = (gz.transpose(2,3)*z0).sum(0)

                # gz: (E,H,1,D)x(E,H,D,K) -> (E,H,1,K)
                try:
                    gz0 = gz @ W
                except:
                    # Compute later if fails due to memory shortage.
                    gz0 = None
            else:
                gz0 = gz

            # p: (E,H,R)
            #p = torch.softmax(x*y, dim=ctx.dim)

            if True:
                a = _get_basis_func(m, K, x[...,None], basis_func, width_scale)
                # a = torch.arange(1,K+1, dtype=x.dtype, device=x.device)[None,None,None]*(m/K)
                # a = a - x**0.5
                # a /= ((2**0.5)*m/K)
                # a *= a
                # torch.neg_(a)
                # torch.exp_(a)
            
                # a: (edges,     1, mirrors, K)
                # x: (edges,     1, mirrors, 1)
                # y: (edges, heads,       1, 1)
                # z: (edges, heads,       1, K)

                # Unloop the last axis (size K) of a
                s = torch.broadcast_shapes(a.shape[:3], gz.shape[:3])
                g2 = torch.zeros(s, device=gz.device, dtype=gz.dtype)
                if gz0 is not None:
                    for i in range(K):
                        g2 += a[...,i]*gz0[...,i]
                else:
                    # Because gz has not been updated as gz@W, compute it here in the loop.
                    # gz@W : (E,H,1,D)x(H,D,K)
                    for i in range(K):
                        g2 += a[...,i]*(gz@W[...,i:i+1]).squeeze(-1)
                del a
                g2 -= (g2*p).sum(dim=ctx.dim, keepdim=True)
                g2 *= p
                del p, gz, gz0
                g += g2

        # gx = y*p*(a*gz - (a*gz*p).sum())
        if ctx.needs_input_grad[0]:
            gx = y*g
            # if x was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gx.shape,x.shape)) if si!=so]
            if len(dims) > 0:
                gx = gx.sum(dim=dims, keepdim=True)

        # gy = x*p*(a*gz - (a*gz*p).sum())
        # (E,H,1,1) = (E,1,R,1)*(E,H,R,1)*(E,H,R,1)
        if ctx.needs_input_grad[1]:
            gy = x*g
            # if y was broadcast in forward, sum the gradients along the broadcast axes.
            dims = [i for i,(si,so) in enumerate(zip(gy.shape,y.shape)) if si!=so]
            if len(dims) > 0:
                gy = gy.sum(dim=dims, keepdim=True)

        return gx, gy, None, None, None, gW, None, None, None

def indexed_lattice_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    batch_q: Tensor,
    batch_kv: Tensor,
    edges: Tensor,
    dist2: Tensor,
    cos_kr: Tensor,
    k2: Tensor,
    vcell: Tensor,
    selfe: Tensor,
    pe_wave: Tensor,
    lattice_pos_weights: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    pe_dist_proj: Optional[Tensor],
    pe_wave_proj: Optional[Tensor],
    training: bool = True,
    need_weights: bool = True,
    gauss_scale: Optional[Tensor] = None,
    atten_scale: Optional[Tensor] = None,
    onehot:Optional[Tensor] = None,
    lattice_params: LatticeformerParams = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        dist: distance matrices of points of lattices.
        lattice_pos_weights: weights for lattice position embeddings.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - dist: :math:`(N, L, S, R)` or `(L, S, R)`, where N is the batch size, S is the source sequence length, 
           N is the batch size, R is the number of neighbors of the lattice.
        - lattice_pos_weights: :math:`(E)`, where E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    _mha_shape_check(query, key, value, edges, dist2, cos_kr, k2, vcell, selfe, num_heads)

    # set up shape vars
    tgt_len, embed_dim = query.shape
    src_len, _ = key.shape
    esz = edges.shape[1]
    # assert embed_dim == embed_dim_to_check, \
    #     f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    # if isinstance(embed_dim, torch.Tensor):
    #     # embed_dim can be a tensor when JIT tracing
    #     head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    # else:
    #     head_dim = embed_dim // num_heads
    # assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    head_dim_q = q_proj_weight.shape[0] // num_heads
    head_dim_k = k_proj_weight.shape[0] // num_heads
    head_dim_v = v_proj_weight.shape[0] // num_heads
    assert head_dim_q*num_heads == q_proj_weight.shape[0]
    assert head_dim_k*num_heads == k_proj_weight.shape[0]
    assert head_dim_v*num_heads == v_proj_weight.shape[0]
    
    # allow MHA to have different embedding dimensions when separate projection weights are used
    assert key.shape[0] == value.shape[0], \
        f"key's sequence and batch dims {key.shape[0]} do not match value's {value.shape[0]}"
   
    #
    # compute in-projection
    #
    assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
    assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
    assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
    if in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = in_proj_bias.split([q_proj_weight.size(0), k_proj_weight.size(0), v_proj_weight.size(0)])
    q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    #
    # reshape q, k, v for multihead attention
    #
    q = q.contiguous().view(tgt_len, num_heads, head_dim_q)
    k = k.contiguous().view(k.shape[0], num_heads, head_dim_k)
    v = v.contiguous().view(v.shape[0], num_heads, head_dim_v)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # compute attn_weights according to dist 
    # TODO: this stes is the largest bottleneck of latticeformer.
    # dist: (esz, neighbors)
    # q: (tgt_len, num_heads, head_dim)
    # lattice_pos_weights: (num_heads*head_dim)
    if lattice_params is None or lattice_params.domain == "no":
        attn_weights = None
    else:
        domain = lattice_params.domain
        gauss_lb_real = lattice_params.gauss_lb_real
        gauss_lb_reci = lattice_params.gauss_lb_reci
        positive_func = lattice_params.positive_func

        # exp( -pos_func(alpha)*distance^2 )
        if lattice_params.gauss_state.startswith("q"):
            alpha = q.reshape(tgt_len,num_heads,1,head_dim_q)@lattice_pos_weights.reshape(num_heads,head_dim_q,1)
        elif lattice_params.gauss_state == "1":
            ones = torch.ones_like(q)
            alpha = ones.reshape(tgt_len,num_heads,1,head_dim_q)@lattice_pos_weights.reshape(num_heads,head_dim_q,1)
        else:
            # query is original state 'x' before q = Wx
            # lattice_pos_weights's shape is (num_heads, embed_dim)
            alpha = F.linear(query, lattice_pos_weights)

        alpha = alpha.view(tgt_len, num_heads)
        if lattice_params.normalize_gauss:
            scale_available = gauss_scale[0] > 0
            if scale_available:
                scale = gauss_scale[0]
                shift = gauss_scale[1]
            else:
                scale = torch.reciprocal(alpha.detach().std())
                shift = -alpha.detach().mean()
                #print("Computed gauss scale and shift:", scale, shift)
            alpha += shift
            alpha *= scale

            # save the scale only once in the first training step.
            if training and not scale_available:
                torch.nn.init.constant_(gauss_scale[0], scale)
                torch.nn.init.constant_(gauss_scale[1], shift)
                print("Saved gauss scale and shift:", gauss_scale.data)
        
        if positive_func == 'abs':
            func = lambda x,lb: (1-lb)*x.abs() + lb
        elif positive_func.startswith('softplus'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x,lb: F.softplus(x + 1/beta*math.log(math.exp(beta*(1-lb))-1), beta=beta) + lb
        elif positive_func.startswith('exp'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x,lb: (1.0-lb)*torch.exp(x*(beta/(1.0-lb))) + lb
        elif positive_func.startswith('elu'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x,lb: (1.0-lb)*F.elu(x*(beta/(1.0-lb))) + 1.0
        elif positive_func.startswith('sigmoid'):
            beta = float(positive_func.split('=')[1]) if '=' in positive_func else 1.0
            func = lambda x,lb: (1/lb-lb)*F.sigmoid(x*(beta*(1+lb)/(1.0-lb)) + math.log(lb)) + lb
        else:
            raise NotImplementedError(f'Unkown positive_func: {positive_func}')

        # a = alpha
        # c = func(alpha, gauss_lb_real)
        # print("layer\t{:.4f}\t{:.4f}\t{:.4f} | {:.4f}\t{:.4f}\t{:.4f}" .format(a.mean(), a.median(), a.std(), c.mean(), c.median(), c.std()))

        '''
        Saved gauss scale and shift: tensor([ 2.5387e+01, -6.5168e-03], device='cuda:0')
        Saved gauss scale and shift: tensor([3.1654e+01, 5.2495e-04], device='cuda:0')
        Saved gauss scale and shift: tensor([3.0025e+01, 7.2772e-03], device='cuda:0')
        Saved gauss scale and shift: tensor([2.2007e+01, 3.1712e-03], device='cuda:0')
        '''
        if domain == "real":
            alpha = func(alpha, gauss_lb_real)*(-0.5*lattice_params.scale_real**-2)
            attn_weights = compute_real_domain_attn(alpha, dist2, edges[0])
        elif domain == "reci":
            alpha = func(alpha, gauss_lb_reci)*(-0.5*lattice_params.scale_reci**2)
            attn_weights = compute_reci_domain_attn(alpha, cos_kr, k2, vcell, edges[0], batch_q)
        elif domain == "RECI":
            # NOTE: for func_test, use this version.
            alpha = 1/func(alpha, gauss_lb_reci)*(-0.5*lattice_params.scale_reci**2)
            attn_weights = compute_reci_domain_attn(alpha, cos_kr, k2, vcell, edges[0], batch_q)
        elif domain == "multihead":
            a1, a2 = alpha.chunk(2, 1)
            a1 = func(a1, gauss_lb_real)*(-0.5*lattice_params.scale_real**-2)
            a2 = func(a2, gauss_lb_reci)*(-0.5*lattice_params.scale_reci**2)
            attn_weights = torch.cat([
                compute_real_domain_attn(a1, dist2, edges[0]),
                compute_reci_domain_attn(a2, cos_kr, k2, vcell, edges[0], batch_q)
            ], dim=-1)
            alpha = torch.cat([a1, a2], dim=1)
        else:
            raise NotImplementedError(f"Not implmeneted for domain = {domain}.")

    if pe_wave_proj is not None and lattice_params.value_pe_condproj != "no":
        # pe_dist_proj: (98, heads, head_dim, pe_dim)
        # onehot: (N, 98)
        W = onehot @ pe_wave_proj.reshape(pe_wave_proj.shape[0], -1)
        if lattice_params.value_pe_condproj in ("i", "I"):
            W = W[edges[0]]
        elif lattice_params.value_pe_condproj in ("j", "IJ"):
            W = W[edges[1]]
        elif lattice_params.value_pe_condproj in ("ij", "IJ"):
            W = (W[edges[0]]+W[edges[1]])*0.5
        else:
            raise NotImplementedError()
        shape0 = pe_wave_proj.shape
        pe_wave_proj = W.reshape(-1, shape0[1], shape0[2], shape0[3])

    # softmax(dist2*alpha)*pe_wave
    # (edges, 1, R, 1)*(edges, heads, 1, 1)*(edges, 1, R, K)
    values = None
    if pe_wave_proj is not None and domain =="real":
        values = AverageBySoftmaxXYFunc_.apply(
            pe_wave[:, None, :, :],
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_wave_proj
        )
    elif pe_wave_proj is not None and domain in ["reci", "RECI"]:
        values = AverageBySoftmaxXYFunc_.apply(
            pe_wave[:, None, :, :],
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_wave_proj
        )
        # (edges, heads, 1, D) = 
        # (edges, heads, 1, K) x (heads, K, D)
        #V = values.squeeze(2)
        #V = torch.stack([V[:,i] @ pe.t() for i,pe in enumerate(pe_wave_proj.chunk(num_heads, 0))], 1)
        #values = V
    
    if pe_dist_proj is not None and lattice_params.value_pe_condproj != "no":
        # pe_dist_proj: (98, heads, head_dim, pe_dim)
        # onehot: (N, 98)
        W = onehot @ pe_dist_proj.reshape(pe_dist_proj.shape[0], -1)
        if lattice_params.value_pe_condproj in ("i", "I"):
            W = W[edges[0]]
        elif lattice_params.value_pe_condproj in ("j", "IJ"):
            W = W[edges[1]]
        elif lattice_params.value_pe_condproj in ("ij", "IJ"):
            W = (W[edges[0]]+W[edges[1]])*0.5
        else:
            raise NotImplementedError()
        shape0 = pe_dist_proj.shape
        pe_dist_proj = W.reshape(-1, shape0[1], shape0[2], shape0[3])

    if pe_dist_proj is not None and domain in "real":
        dist_max = lattice_params.value_pe_dist_max
        if dist_max < 0:
            dist_max = (-dist_max)*lattice_params.scale_real
        values = AverageGaussiansBySoftmaxXYFunc_.apply(
            dist_max,
            lattice_params.value_pe_dist_real,
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_dist_proj,
            lattice_params.value_pe_width_scale
        )
    elif pe_dist_proj is not None and domain in ["reci", "RECI"]:
        dist_max = lattice_params.value_pe_dist_max
        if dist_max < 0:
            dist_max = (-dist_max)*lattice_params.scale_real
        values = AverageGaussiansBySoftmaxXYFunc_.apply(
            dist_max,
            lattice_params.value_pe_dist_reci,
            dist2[:, None, :, None],
            alpha[:, :, None, None][edges[0]],
            -2,
            pe_dist_proj,
            lattice_params.value_pe_width_scale
        )
        #values = values @ pe_wave_proj.view(num_heads, head_dim, -1).transpose(1,2)
        #values = values.view(-1, num_heads, head_dim)
        
    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights, sizes = _scaled_dot_product_attention(q, k, v, batch_q, batch_kv, edges, attn_weights, values, dropout_p, lattice_params.norm_func_mode)
    
    if lattice_params.norm_func_mode < 0:
        scale_available = atten_scale[0] > 0
        if scale_available:
            scale = atten_scale[0]
        else:
            qsize, ksize, esize = sizes
            scale = attn_output_weights.detach()
            scale.abs_()
            scale = torch.split_with_sizes(scale, esize)
            scale = torch.stack([x.view(qs,ks,-1).sum(dim=1).mean() for x,qs,ks in zip(scale, qsize, ksize)])
            #print(scale)
            scale = scale.mean().reciprocal()
        attn_output *= scale

        # save the scale only once in the first training step.
        if training and not scale_available:
            torch.nn.init.constant_(atten_scale[0], scale)
            print("Saved atten scale:", atten_scale.data)

    attn_output = attn_output.contiguous().view(tgt_len, head_dim_v*num_heads)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(esz, num_heads)
        attn_output_weights = attn_output_weights.mean(dim=1)
        return attn_output, attn_output_weights
    else:
        return attn_output, values

class IndexedLatticeMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, 
                 kdim=None, vdim=None, 
                 params=LatticeformerParams(),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(IndexedLatticeMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim*num_heads if kdim is not None and kdim>0 else embed_dim
        self.vdim = vdim*num_heads if vdim is not None and vdim>0 else embed_dim
        assert params.domain in ("real", "reci", "RECI", "multihead", "no")
        if params.domain == "multihead":
            assert num_heads % 2 == 0, "In multihead mode, num_head must be even as MHA of each domain uses num_head/2 of heads."
        
        # lattice parameters that are referenced in indexed_lattice_multi_head_attention_forward.
        self.params = params
        self.gauss_scale = Parameter(torch.zeros(2, **factory_kwargs), requires_grad=False)
        self.atten_scale = Parameter(torch.zeros(1, **factory_kwargs), requires_grad=False)

        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((self.vdim, embed_dim), **factory_kwargs))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(self.kdim*2+self.vdim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(self.vdim, embed_dim, bias=bias, **factory_kwargs)

        if params.domain == "no":
            self.lattice_pos_weights = None
        elif params.gauss_state.startswith("q"):
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state == "1":
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state == "x-xn":
            self.lattice_pos_weights = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
            self.lattice_pos_weights2 = Parameter(torch.empty((self.kdim), **factory_kwargs))
        elif params.gauss_state.startswith("x"):
            self.lattice_pos_weights = Parameter(torch.empty((self.num_heads, embed_dim), **factory_kwargs))
            
        self.ATOM_NUM = 98
        head = self.num_heads if params.value_pe_headed else 1
        cond = 1 if params.value_pe_condproj=="no" else self.ATOM_NUM

        if params.domain in ["real", "multihead"]:
            self.pe_dist_proj = Parameter(torch.empty(cond, head, self.head_dim, params.value_pe_dist_real, **factory_kwargs)) \
                if params.value_pe_dist_real > 0 else None
            self.pe_wave_proj = Parameter(torch.empty(cond, head, self.head_dim, params.value_pe_wave_real**3, **factory_kwargs)) \
                if params.value_pe_wave_real > 0 else None
        elif params.domain in ["reci", "RECI"]:
            self.pe_dist_proj = Parameter(torch.empty(cond, head, self.head_dim, params.value_pe_dist_reci, **factory_kwargs)) \
                if params.value_pe_dist_reci > 0 else None
            self.pe_wave_proj = Parameter(torch.empty(cond, head, self.head_dim, params.value_pe_wave_reci**3, **factory_kwargs)) \
                if params.value_pe_wave_reci > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.out_proj.weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

        # TODO: consider the normalization method later
        if self.lattice_pos_weights is not None:
            if self.params.gauss_state in ("q", "q-headdim", "1"):
                normal_(self.lattice_pos_weights, 0., (self.kdim//self.num_heads)**-0.5)
            elif self.params.gauss_state == "q-kdim":
                normal_(self.lattice_pos_weights, 0., (self.kdim)**-0.5)
            elif self.params.gauss_state in ("x", "x-norm"):
                normal_(self.lattice_pos_weights, 0., (self.embed_dim)**-0.5)
            elif self.params.gauss_state == "x-xavier":
                xavier_uniform_(self.lattice_pos_weights)
            elif self.params.gauss_state == "x-xn":
                xavier_uniform_(self.lattice_pos_weights)
                normal_(self.lattice_pos_weights2, 0., (self.kdim//self.num_heads)**-0.5)
            elif self.params.gauss_state == "x-xn2":
                H, K, D = self.num_heads, self.kdim, self.embed_dim
                W = self.lattice_pos_weights
                W1 = torch.empty((K, D), device=W.device, dtype=W.dtype)
                W2 = torch.empty((H,1,K//H), device=W.device, dtype=W.dtype)
                xavier_uniform_(W1)
                normal_(W2, 0., (K//H)**-0.5)
                W0 = W2 @ W1.reshape((H, K//H, D))
                with torch.no_grad():
                    self.lattice_pos_weights.set_(W0.reshape_as(W))
            else:
                raise NotImplementedError()

        # D0 = 1/sqrt(I + O)
        # D1 = 1/sqrt(I + O*H)
        # I = 64, O = 16, H = 8
        # D0 = 1/sqrt(64 + 16) = 1/( 4*sqrt(5) )
        # D1 = 1/sqrt(64 + 16*8) = 1/( 8*sqrt(3) )
        # D1/D0 = 0.645497224
        # good case = 0.0099
        # current (no t-fixup)  = 0.0063
        if self.pe_dist_proj is not None:
            with torch.no_grad():
                for i, W in enumerate(self.pe_dist_proj):
                    if i != 0 and self.params.value_pe_condproj in ("I", "J", "IJ"):
                        W.set_(self.pe_dist_proj[0])
                    else:
                        W = W.view(-1, W.shape[-1])
                        co, ci = W.shape
                        a = (self.embed_dim + ci) / (co + ci)
                        # 'a' is to keep the scale regardless of pe_headed True or False.
                        xavier_uniform_(W, (ci)**-0.5)

        if self.pe_wave_proj is not None:
            with torch.no_grad():
                for i, W in enumerate(self.pe_wave_proj):
                    if i != 0 and self.params.value_pe_condproj in ("I", "J", "IJ"):
                        W.set_(self.pe_wave_proj[0])
                    else:
                        W = W.view(-1, W.shape[-1])
                        co, ci = W.shape
                        a = (self.embed_dim + ci) / (co + ci)
                        # 'a' is to keep the scale regardless of pe_headed True or False.
                        xavier_uniform_(W, (ci)**-0.5)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, batch_q:Tensor, batch_kv:Tensor, edges: Tensor,
                dist2: Tensor=None, 
                cos_kr: Tensor=None, k2: Tensor=None, vcell: Tensor=None, selfe:Tensor=None, pe_wave: Tensor=None,
                onehot:Tensor=None,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """

        W = self.lattice_pos_weights
        if self.params.gauss_state == "x-xn":
            H, K, D = self.num_heads, self.kdim, self.embed_dim
            W2 = self.lattice_pos_weights2.view(H, 1, K//H)
            W = W2 @ W.reshape((H, K//H, D))
            W = W.reshape(H, D)

        attn_output, attn_output_weights = indexed_lattice_multi_head_attention_forward(
            query, key, value, batch_q, batch_kv, edges, 
            dist2, 
            cos_kr, k2, vcell, selfe, pe_wave,
            W,
            self.embed_dim, self.num_heads,
            self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
            self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            self.pe_dist_proj,
            self.pe_wave_proj,
            training=self.training,
            need_weights=need_weights,
            gauss_scale=self.gauss_scale,
            atten_scale=self.atten_scale,
            onehot=onehot,
            lattice_params=self.params)
        return attn_output, attn_output_weights